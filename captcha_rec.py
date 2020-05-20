import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from model import make_model
from default_config import Config
from dataset import FakeDataset as dataset, num_classes, label_length


def make_dataloader(is_train, config):
    dataset_size = num_classes ** label_length if is_train else num_classes ** 2
    return DataLoader(
        dataset(num_classes ** 2),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def train(config: Config):
    print(
        f"Watch tensorboard log file for training details at position {config.log_file}"
    )
    model = make_model(
        "resnet50", [3, 4, 6, 3], num_classes, label_length, False
    ).cuda()
    dataloader = make_dataloader(True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    ctc_loss = torch.nn.funcional.ctc_loss

    loss, acc = 0, 0
    for ibatch, (image, target, target_str) in enumerate(dataloader):
        image, target = image.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(image)
        # torch.fill 一定要用image.shape[0] 而不是用config.batch_size
        loss = ctc_loss(
            output,
            target,
            torch.fill((image.shape[0],), label_length, dtype=torch.long),
            torch.fill((image.shape[0],), label_length, dtype=torch.long),
        )

        if ibatch != 0 and ibatch % config.print_step == 0:
            pass
        exit(0)
