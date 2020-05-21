# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py
from default_config import Config
import argparse
import torch
from captcha_rec import train

if __name__ == "__main__":
    config = Config()

    if config.cuda:
        parser = argparse.ArgumentParser()
        torch.distributed.init_process_group(backend="nccl")
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)
        config.local_rank = args.local_rank

    train(config)
