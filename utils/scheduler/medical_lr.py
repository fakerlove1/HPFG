import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim import lr_scheduler


class Medical_LR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_iterations):
        self.base_lr = base_lr
        self.max_iterations = max_iterations
        super(Medical_LR, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        iter_num = self.last_epoch-1
        current_lr = self.base_lr * \
            (1.0 - iter_num / self.max_iterations) ** 0.9
        return [current_lr for _ in self.base_lrs]