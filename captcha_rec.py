import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from model import make_model
from default_config import Config
from dataset import FakeDataset as dataset, num_classes, label_length, net_output_right
import time


def make_dataloader(is_train, config):
    dataset_size = num_classes ** label_length * 100 if is_train else num_classes ** 2
    dset = dataset(dataset_size)
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
    )


def train(config: Config):
    if config.local_rank == 0:
        print(
            f"Watch tensorboard log file for training details at position {config.log_file}"
        )
    if config.local_rank == 0:
        tx_writer = tensorboard.SummaryWriter(config.log_file)

    model = make_model(
        "resnet50", [3, 4, 6, 3], num_classes, label_length, False
    ).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,device_ids=[config.local_rank],output_device=config.local_rank
    )
    model.train()

    dataloader = make_dataloader(True,config)
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e6, gamma=0.99)
    ctc_loss = torch.nn.CTCLoss()

    losses, acc_es = 0, 0
    time_record = time.time()
    for ibatch, (image, target, target_str) in enumerate(dataloader):
        image, target = image.cuda(), target.cuda()

        optimizer.zero_grad()

        # output.shape = [input_length,batch_size,num_classes]
        output = model(image)
        output = output.permute(2,0,1)

        # torch.fill 一定要用image.shape[0] 而不是用config.batch_size
        loss = ctc_loss(
            output,
            target,
            torch.full((image.shape[0],), label_length, dtype=torch.long),
            torch.full((image.shape[0],), label_length, dtype=torch.long),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses += loss.item()
        acc_es += net_output_right(output,target)
        best_score = 0

        if ibatch != 0 and ibatch % config.print_step == 0:
            curr_los = losses / config.print_step
            curr_acces = acc_es / config.print_step

            if curr_acces > best_score:
                torch.save(model.state_dict(),"models_saved/ctc_loss_best.pth")
                best_score = curr_acces

            if config.local_rank == 0:
                print('Image %d; losses: %.3f; acc: %.3f; Time Cost: %.1f'  % (ibatch * config.batch_size,curr_los,curr_acces,time.time() - time_record ))
                tx_writer.add_scalar("Loss/train",curr_los,ibatch)
                tx_writer.add_scalar("Acc/train",curr_acces,ibatch)
                tx_writer.add_scalar("Time/train",time.time() - time_record,ibatch)
                torch.save(model.state_dict(),"models_saved/ctc_loss.pth")
            losses,acc_es = 0,0
            time_record = time.time()
