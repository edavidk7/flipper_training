from torch.optim.lr_scheduler import LRScheduler


class LinearDecayThenConstantLR(LRScheduler):
    def __init__(self, optimizer, end_factor: float = 0.0, end_step: int = 0, last_epoch: int = -1):
        self.end_factor = end_factor
        self.end_step = end_step
        super(LinearDecayThenConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        coef = 1 - (1 - self.end_factor) * min(self.last_epoch, self.end_step) / self.end_step
        return [coef * lr for lr in self.base_lrs]
