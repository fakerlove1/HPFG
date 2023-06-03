import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import random

from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables
from utils import build_lr_scheduler, build_optimizer
from model import build_model
from datasets import build_loader
from val import test_isic

# paper link https://arxiv.org/abs/2202.00677


def main():
    path = r"config/ict-medseg_unet_30k_224x224_ISIC.yaml"
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    label_loader, unlabel_loader, test_loader, = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> label_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = build_model(args=args).to(device=args.device)  # 创建模型
    # ema_model = deepcopy(model)  # 创建ema_model
    ema_model = build_model(args=args).to(device=args.device)
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # step 2: 训练模型
    ICT_MedSeg(model, ema_model, label_loader, unlabel_loader, test_loader, args)


def ICT_MedSeg(model, ema_model, label_loader, unlabel_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)

    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    model.train()
    cur_itrs = 0
    train_loss = 0
    best_dice = 0.0
    best_ema_dice = 0.0

    #  加载原模型
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        cur_itrs = state_dict["cur_itrs"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["lr_scheduler"]
        best_dice = state_dict["best_score"]

    label_iter = iter(label_loader)

    for epoch in range(max_epoch):
        train_loss = 0
        for i, (img_unlabeled, _) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1
            try:
                img_labeled, target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                img_labeled, target_label, = next(label_iter)

            img_labeled = img_labeled.to(args.device).float()
            target_label = target_label.to(args.device).long()
            img_unlabeled = img_unlabeled.to(args.device).float()

            label_bs = img_labeled.shape[0]
            unlabel_bs = img_unlabeled.shape[0]

            # ICT mix factors
            ict_mix_factors = np.random.beta(args.ict_alpha, args.ict_alpha, size=(unlabel_bs // 2, 1, 1, 1))
            ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).cuda()

            unlabeled_volume_batch_0 = img_unlabeled[0:unlabel_bs // 2, ...]
            unlabeled_volume_batch_1 = img_unlabeled[unlabel_bs // 2:, ...]
            batch_ux_mixed = unlabeled_volume_batch_0 * (1.0 - ict_mix_factors) + unlabeled_volume_batch_1 * ict_mix_factors

            input_volume_batch = torch.cat([img_labeled, batch_ux_mixed], dim=0)

            outputs = model(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output_ux0 = torch.softmax(ema_model(unlabeled_volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(ema_model(unlabeled_volume_batch_1), dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors

            loss_ce = criterion(outputs[:label_bs], target_label)
            loss_dice = dice_loss(outputs_soft[:label_bs], target_label.unsqueeze(1))

            supervised_loss = 0.6 * loss_dice + 0.4 * loss_ce

            consistency_weight = get_current_consistency_weight(epoch=cur_itrs // 150, args=args)
            consistency_loss = torch.mean((outputs_soft[label_bs:] - batch_pred_mixed) ** 2)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            update_ema_variables(model, ema_model, args.ema_decay, cur_itrs)

            train_loss += loss.item()
            args.writer.add_scalar('ICT_MedSeg/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('ICT_MedSeg/lr', lr, cur_itrs)
            args.writer.add_scalar('ICT_MedSeg/consistency_weight', consistency_weight, cur_itrs)
            args.writer.add_scalar('ICT_MedSeg/consistency_loss', consistency_loss, cur_itrs)

            if cur_itrs % args.step_size == 0:
                dice, hd95 = test_isic(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="test_model")
                args.writer.add_scalar('ICT_MedSeg/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('ICT_MedSeg/{}_hd95'.format(args.name), hd95, cur_itrs)
                args.logger.info("epoch:{} \t model dice:{:.4f} \t hd95:{:.4f} ".format(epoch, dice, hd95))

                if dice > best_dice:
                    best_dice = dice
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, args.model_save_path)

                dice, hd95 = test_isic(model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="test_ema_model")

                if dice > best_ema_dice:
                    best_ema_dice = dice
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice,
                        "model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, args.ema_model_save_path)

                args.writer.add_scalar('ICT_MedSeg/{}_dice_ema'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('ICT_MedSeg/{}_hd95_ema'.format(args.name), hd95, cur_itrs)
                args.logger.info("epoch:{} \t ema_model dice:{:.4f} \t hd95:{:.4f} ".format(epoch, dice, hd95))
                args.logger.info("best model dice:{:.4f} ,best ema_model dice {:.4f}".format(best_dice, best_ema_dice))

                model.train()
                ema_model.train()
            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.4f} ".format(cur_itrs, args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss
                                                                          ))


if __name__ == '__main__':
    main()
