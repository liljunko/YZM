from default_config import Config
import argparse
import torch
from captcha_rec import train

if __name__ == "__main__":
    config = Config()
    config.num_workers = 0
    config.batch_size = 1
    config.local_rank = 0
    config.log_file = "./log/debug"
    assert config.cuda, "Please don't train your code using only cpu"

    train(config)

