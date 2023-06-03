import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import random

from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss
from model import build_model
from datasets import build_loader
from val import test_isic


def main():
    
    path = r"config/cps_unet_30k_224x224_ISIC.yaml"
    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args = loadyaml(os.path.join(root, path))  # 加载yaml
    if args.cuda:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args.save_path = os.path.join(root, args.save_path)
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.model1_save_path = os.path.join(args.save_path, "model", "model1.pth")
    args.model2_save_path = os.path.join(args.save_path, "model", "model2.pth")

    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log")
    torch.manual_seed(args.seed)  # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    label_loader, unlabel_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(label_loader) * args.batch_size))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader) * args.unlabel_batch_size))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model1 = build_model(args=args.model1).to(device=args.device)  # 创建模型
    model2 = build_model(args=args.model2).to(device=args.device)  # 创建模型

    # step 2: 训练模型
    CPS(model1, model2, label_loader, unlabel_loader, test_loader, args)


def CPS(model1, model2, label_loader, unlabel_loader, test_loader, args):
    optimizer1 = build_optimizer(args=args.model1, model=model1)
    lr_scheduler1 = build_lr_scheduler(args=args.model1, optimizer=optimizer1)

    optimizer2 = build_optimizer(args=args.model2, model=model2)
    lr_scheduler2 = build_lr_scheduler(args=args.model2, optimizer=optimizer2)

    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    med_loss = Med_Sup_Loss(args.num_classes)

    model1.train()
    model2.train()
    cur_itrs = 0

    best_dice1 = 0.0
    best_dice2 = 0.0
    args.logger.info("max epoch: {}".format(max_epoch))
    args.logger.info("start training")
    label_iter = iter(label_loader)
    for epoch in range(max_epoch):
        train_loss = 0.0
        for i, (unlabel_img, _) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1
            try:
                label_img, target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                label_img, target_label, = next(label_iter)

            label_img = label_img.to(args.device).float()
            unlabel_img = unlabel_img.to(args.device).float()
            target_label = target_label.to(args.device).long()
            label_bs = label_img.shape[0]

            x = torch.cat([label_img, unlabel_img], dim=0)
            x = x.to(args.device).float()
            output1 = model1(x)
            output2 = model2(x)
            output_soft1 = torch.softmax(output1, dim=1)
            output_soft2 = torch.softmax(output2, dim=1)

            # step 3: 计算损失
            # sup loss
            loss_sup1 = med_loss(output1[:label_bs], target_label)
            loss_sup2 = med_loss(output2[:label_bs], target_label)
            loss_sup = loss_sup1 + loss_sup2
            # semi_loss
            pseudo_label1 = torch.argmax(output_soft1[label_bs:].detach(), dim=1, keepdim=False)
            pseudo_label2 = torch.argmax(output_soft2[label_bs:].detach(), dim=1, keepdim=False)
            loss_semi = med_loss(output1[label_bs:], pseudo_label2)+med_loss(output2[label_bs:], pseudo_label1)
            consistency_weight = get_current_consistency_weight(epoch=cur_itrs // 150, args=args)
            loss = loss_sup + consistency_weight * loss_semi

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            lr_scheduler1.step()
            lr_scheduler2.step()
            train_loss += loss.item()

            lr = optimizer1.param_groups[0]["lr"]
            args.writer.add_scalar('mynet/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('mynet/lr', lr, cur_itrs)
            args.writer.add_scalar('mynet/consistency_weight', consistency_weight, cur_itrs)
            args.writer.add_scalar('mynet/loss_semi', loss_semi.item(), cur_itrs)
            args.writer.add_scalar('mynet/loss_sup', loss_sup.item(), cur_itrs)

            if cur_itrs % args.step_size == 0:
                mean_dice, mean_hd952 = test_isic(model=model1, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                args.logger.info("model1 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mynet/model1_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mynet/model1_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice1:
                    best_dice1 = mean_dice
                    torch.save(
                        {
                            "model": model1.state_dict(),
                            "optimizer": optimizer1.state_dict(),
                            "lr_scheduler": lr_scheduler1.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.model1_save_path)

                mean_dice, mean_hd952 = test_isic(model=model2, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model2")
                args.logger.info("model2 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mynet/model2_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mynet/model2_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice2:
                    best_dice2 = mean_dice
                    torch.save(
                        {
                            "model": model2.state_dict(),
                            "optimizer": optimizer2.state_dict(),
                            "lr_scheduler": lr_scheduler2.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.model2_save_path)
                args.logger.info("model1 best_dice: {:.4f}, model2 best_dice: {:.4f}".format(best_dice1, best_dice2))
                model1.train()
                model2.train()
            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs, args.total_itrs,
                         100. * cur_itrs / args.total_itrs, train_loss / len(unlabel_loader)))


if __name__ == '__main__':
    main()
