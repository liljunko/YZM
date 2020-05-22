class Config:
    lr = 1e-3
    cuda = True
    batch_size = 256
    num_workers = 64
    print_step = 128
    log_file = "./logs/first_logs/"
    model_resume = None
    model_saved = "models_saved/ctc_loss.pth"
    local_rank = 0  # only for intelliense, useless otherwise
