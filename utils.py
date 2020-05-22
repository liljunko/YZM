import torch
import torch.utils.tensorboard as tensorboard


class RankWorker(object):
    def __init__(self, config: Config, rank_print=0):
        """
        only work at specific local_rank process
        """
        super().__init__()
        self.config = config
        self.isWork = (rank_print == self.config.local_rank)
        # ANCHOR tensorboard init
        if self.isWork:
            self.tx_writer = tensorboard.SummaryWriter(config.log_file)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.isWork:
            self.tx_writer.add_scalar(tag, scalar_value, global_step, walltime)

    def print(self, **kargs):
        if self.isWork:
            print(**kargs)

    def do(self, function, **kargs):
        if self.isWork:
            return function(**kargs)

    def save(self, filename, dict_saved):
        if self.isWork:
            torch.save(dict_saved, filename)