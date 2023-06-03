import os.path
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from utils import loadyaml, _get_logger, mk_path, Med_Sup_Loss
from model import build_model
from datasets import build_loader
from utils import build_lr_scheduler, build_optimizer
# from val import test_acdc
import math
from torchvision.utils import make_grid

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler


def main():

    path = r"config/swinmae_30k_224x224_ACDC.yaml"
    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args = loadyaml(os.path.join(root, path))  # 加载yaml

    if args.cuda:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    torch.manual_seed(args.seed)  # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args.save_path = os.path.join(root, args.save_path)
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.finetune_save_path = os.path.join(args.save_path, "model", "finetune_model.pth")
    args.pretrain_save_path = os.path.join(args.save_path, "model", "pretrain_model.pth")
    args.supervise_save_path = os.path.join(args.save_path, "model", "supervise_model.pth")  # 设置模型名称

    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log")

    # step 1: 构建数据集
    train_loader, test_loader = build_loader(args)
    args.epochs = args.total_itrs // len(train_loader) + 1
    args.logger.info("==========> train_loader length:{}".format(len(train_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 2: 构建模型
    model = build_model(args=args).to(device=args.device)

    # step 3: 训练模型
    Supervise(model=model, train_loader=train_loader, test_loader=test_loader, args=args)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def Supervise(model: nn.Module, train_loader, test_loader, args):
    # # optimizer = build_optimizer(args=args, model=model)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))
    # # lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    max_epoch = args.total_itrs // len(train_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # criterion=BCEDiceLoss()
    # criterion=Med_Sup_Loss(args.num_classes)

    model.train()
    cur_itrs = 0
    train_loss = 0.0
    best_dice = 0.0

    #  加载原模型
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"], strict=True)

    for epoch in range(num_epochs):
        for i, (img, label_true) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
            img = img.to(args.device).float()
            label_true = label_true.to(args.device).long()
            predicted_img, mask = model(img)
            # loss=loss.mean()
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=epoch)
            # adjust_learning_rate(optimizer=optimizer, epoch=epoch, args=args)
            lr = optimizer.param_groups[0]["lr"]
            train_loss += loss.item()
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)

            if cur_itrs % args.step_size == 0:
                model.eval()
                y, mask = model(img)
                # y = model.unpatchify(y)
                y = torch.einsum('nchw->nhwc', y).detach().cpu()

                mask = mask.detach()
                # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size ** 2 * args.in_channels)  # (N, H*W, p*p*3)
                # mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

                x = torch.einsum('nchw->nhwc', img).detach().cpu()
                # masked image
                im_masked = x * (1 - mask)
                # y = y * mask
                # MAE reconstruction pasted with visible patches
                im_paste = x * (1 - mask) + y * mask

                image = make_grid(tensor=img, nrow=4, normalize=True, scale_each=True)
                im_masked = make_grid(tensor=im_masked.permute(0, 3, 1, 2), nrow=4, normalize=True, scale_each=True)
                im_paste = make_grid(tensor=im_paste.permute(0, 3, 1, 2), nrow=4, normalize=True, scale_each=True)

                args.writer.add_image('SwinMae/image', image, cur_itrs)
                args.writer.add_image('SwinMae/im_masked', im_masked, cur_itrs)
                args.writer.add_image('SwinMae/im_paste', im_paste, cur_itrs)
                model.train()

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
        }, args.supervise_save_path)

        # if cur_itrs > args.total_itrs:
        #     return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f}\t  ".format(cur_itrs, args.total_itrs,
                                                                             100. * cur_itrs / args.total_itrs,
                                                                             train_loss/len(train_loader)))
        train_loss = 0


if __name__ == "__main__":
    main()
