from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self):
        self.logger = SummaryWriter()
        self.cnt = 0

    def log(self, d):
        for k,v in d.items():
            self.logger.add_scalar(k, v, self.cnt)
        self.cnt += 1

