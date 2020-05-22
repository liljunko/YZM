import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from model import make_model
from default_config import Config
from dataset import FakeDataset as fake_dataset
from dataset import num_classes, label_length, abs_acc, mean_acc
from utils import RankWorker
import time


def make_dataloader(is_train, config):
    dataset_size = num_classes**label_length * 100 if is_train else num_classes**2
    dset = fake_dataset(dataset_size)
    return DataLoader(dset,
                      batch_size=config.batch_size,
                      num_workers=config.num_workers,
                      pin_memory=True)


def train(config: Config):
    rank_worker = RankWorker(config)
    rank_worker.print(
        f"Watch tensorboard log file for training details at position {config.log_file}"
    )
    begin_batch = 0

    # ANCHOR: build model and init model's state_dict
    model = make_model("resnet50", [3, 4, 6, 3], num_classes, label_length,
                       False)
    if config.cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[config.local_rank],
            output_device=config.local_rank)

    if config.model_resume is not None:
        checkpoint = torch.load(config.model_saved)
        model.load_state_dict(checkpoint['net_state_dict'])
        begin_batch = checkpoint['iter_batch']

    # ANCHOR: dataloader,optimizer,scheduler,loss settings
    dataloader = make_dataloader(True, config)
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1e6,
                                                gamma=0.99)
    ctc_loss = torch.nn.CTCLoss()

    # ANCHOR: losses and acces recorder
    losses, abs_acces, mean_acces = 0, 0, 0
    time_record = time.time()

    # ANCHOR: train loop
    model.train()
    best_score = 0
    for _ibatch, (image, target, target_str) in enumerate(dataloader):
        ibatch = begin_batch + _ibatch
        if config.cuda:
            image, target = image.cuda(), target.cuda()

        optimizer.zero_grad()

        # output4loss.shape -> [input_length,batch_size,num_classes]
        output = model(image)
        output4loss = output.permute(2, 0, 1)

        # torch.fill 一定要用image.shape[0] 而不是用config.batch_size
        loss = ctc_loss(
            output4loss,
            target,
            torch.full((image.shape[0], ), label_length, dtype=torch.long),
            torch.full((image.shape[0], ), label_length, dtype=torch.long),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        predict = torch.argmax(output.detach(), dim=2)

        # ANCHOR: update losses and acces record
        curr_lr = scheduler.get_lr()[0]
        losses += loss.item()
        abs_acces += abs_acc(predict, target)
        mean_acces += mean_acc(predict, target)

        # ANCHOR: print statistic and save model
        if ibatch != 0 and ibatch % config.print_step == 0:
            curr_los = losses / config.print_step
            curr_acces = abs_acces / config.print_step
            curr_meanacc = mean_acces / config.print_step

            if curr_acces > best_score:
                torch.save(model.state_dict(),
                           "models_saved/ctc_loss_best.pth")
                best_score = curr_acces

            rank_worker.print(
                'Image %d; losses: %.3f; [Abs,mean]acc: [%.3f, %.3f]; Time Cost: %.1f; Lr=%.3f'
                % (ibatch * config.batch_size, curr_los, curr_acces,
                   curr_meanacc, time.time() - time_record, curr_lr))
            rank_worker.add_scalar("Loss/train", curr_los, ibatch)
            rank_worker.add_scalar("Acc/train", curr_acces, ibatch)
            rank_worker.add_scalar("Mean Acc/train", curr_meanacc, ibatch)
            rank_worker.add_scalar("Time/train",
                                   time.time() - time_record, ibatch)
            rank_worker.add_scalar("Learning Rate/train", curr_lr, ibatch)
            rank_worker.save(
                {
                    "net_state_dict": model.state_dict(),
                    "iter_batch": ibatch
                }, config.model_saved)

            losses, abs_acces = 0, 0
            time_record = time.time()
