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
from val import test_acdc


def main():
    
    path = r"config/mean_teacher_unet_100_30k_224x224_ACDC.yaml"
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
    args.model_save_path = os.path.join(args.save_path, "model", "model.pth")
    args.ema_model_save_path = os.path.join(args.save_path, "model", "ema_model_model.pth")

    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log")
    torch.manual_seed(args.seed)  # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    train_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(train_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = build_model(args=args).to(device=args.device)  # 创建模型
    ema_model = deepcopy(model)  # 创建ema_model
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # step 2: 训练模型
    Mean_Teacher(model, ema_model, train_loader, test_loader, args)


def Mean_Teacher(model, ema_model, train_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)
    max_epoch = args.total_itrs // len(train_loader) + 1
    med_loss = Med_Sup_Loss(args.num_classes)

    model.train()
    ema_model.train()
    cur_itrs = 0

    best_dice1 = 0.0
    best_dice2 = 0.0

    args.logger.info("max epoch: {}".format(max_epoch))
    args.logger.info("start training")
    for epoch in range(max_epoch):
        train_loss = 0.0
        for i, (label_img, target_label) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
      
            label_img = label_img.to(args.device).float()
            target_label = target_label.to(args.device).long()
            label_bs = label_img.shape[0]
            output = model(label_img)
            output_soft = torch.softmax(output, dim=1)

            with torch.no_grad():
                ema_output = ema_model(label_img)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_sup = med_loss(ema_output_soft, target_label)
            loss_consistence = torch.mean((output_soft - ema_output_soft) ** 2)
            consistency_weight = get_current_consistency_weight(epoch=cur_itrs // 150, args=args)
            loss = loss_sup + consistency_weight * loss_consistence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            update_ema_variables(model, ema_model, args.ema_decay, cur_itrs)

            args.writer.add_scalar('mean_teacher/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('mean_teacher/lr', lr, cur_itrs)
            args.writer.add_scalar('mean_teacher/consistency_weight', consistency_weight, cur_itrs)

            if cur_itrs % args.step_size == 0:
                mean_dice, mean_hd952 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                args.logger.info("model1 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mean_teacher/model1_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mean_teacher/model1_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice1:
                    best_dice1 = mean_dice
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.model_save_path)

                mean_dice, mean_hd952 = test_acdc(model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model2")
                args.logger.info("model2 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mean_teacher/model2_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mean_teacher/model2_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice2:
                    best_dice2 = mean_dice
                    torch.save(
                        {
                            "model": ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.ema_model_save_path)

                args.logger.info(
                    "model1 best_dice: {:.4f}, model2 best_dice: {:.4f}".
                    format(best_dice1, best_dice2))
                model.train()
                ema_model.train()

            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs, args.total_itrs,
                         100. * cur_itrs / args.total_itrs, train_loss / len(unlabel_loader)))


if __name__ == '__main__':
    main()
