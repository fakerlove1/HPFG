import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim import lr_scheduler


class CosineWarmupLR_Scheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs=10, warmup_lr=1e-6,
                 num_epochs=100, base_lr=0.01, final_lr=1e-6, iter_per_epoch=1000):
        """
        学习率设置
        :param optimizer: 优化器
        :param warmup_epochs: 热身epoch,
        :param warmup_lr: 热身学习率
        :param num_epochs: 一共的epoch
        :param base_lr: 基础学习率
        :param final_lr: 最后学习率
        """
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)+1
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        super(CosineWarmupLR_Scheduler, self).__init__(
            optimizer, last_epoch=-1)

    def get_lr(self):
        # if self.last_epoch < self.warmup_epochs:
        #     alpha = self.last_epoch / self.warmup_epochs
        #     return self.base_lr + (self.final_lr - self.base_lr) * alpha
        # else:
        #     return self.base_lr
        # print(self.last_epoch)
        current_lr = self.lr_schedule[self.last_epoch-1]
        return [current_lr for _ in self.base_lrs]
