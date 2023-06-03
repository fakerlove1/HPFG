import os

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler


class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class Cosine_LR_Scheduler(object):
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
        :param iter_per_epoch: 每个epoch的iter
        """

        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


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
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
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
