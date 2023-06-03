import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import random

from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables,linear_rampup
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss
from model import build_model
from datasets import build_loader
from val import test_acdc

# paper link https://arxiv.org/abs/2208.06449


def main():

    path = r"config/s4cvnet_unet_30k_224x224_ACDC.yaml"
    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args = loadyaml(os.path.join(root, path))  # 加载yaml

    if args.cuda:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    args.save_path = os.path.join(root, args.save_path)
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.model1_save_path = os.path.join(args.save_path, "model", "model1.pth")
    args.model2_save_path = os.path.join(args.save_path, "model", "model2.pth")
    args.ema_model_save_path = os.path.join(args.save_path, "model", "ema_model.pth")  # 设置模型名称

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
    args.logger.info("==========> train_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model1 = build_model(args=args.model1).to(device=args.device)  # 创建模型1,一般是unet
    model2 = build_model(args=args.model2).to(device=args.device)  # 创建模型2,一般是swinunet

    ema_model = deepcopy(model2)  # 创建ema_model
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    ema_model.to(device=args.device)

    # step 2: 训练模型
    S4CVnet(model1, model2, ema_model, label_loader, unlabel_loader, test_loader, args)


def S4CVnet(model1, model2, ema_model, label_loader, unlabel_loader, test_loader, args):

    optimizer1 = build_optimizer(args=args.model1, model=model1)
    lr_scheduler1 = build_lr_scheduler(args=args.model1, optimizer=optimizer1)

    optimizer2 = build_optimizer(args=args.model2, model=model2)
    lr_scheduler2 = build_lr_scheduler(args=args.model2, optimizer=optimizer2)

    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    model1.train()
    model2.train()
    cur_itrs = 0
    best_dice1 = 0.0
    best_dice2 = 0.0
    best_ema_dice=0.0
    label_iter = iter(label_loader)

    try:
        pbar = tqdm(total=args.total_itrs)
        for epoch in range(max_epoch):
            run_loss = 0.0
            for idx, (img_unlabel, _) in enumerate(unlabel_loader):
                cur_itrs += 1
                try:
                    img_labeled, target_label = next(label_iter)
                except StopIteration:
                    label_iter = iter(label_loader)
                    img_labeled, target_label, = next(label_iter)

                target_label = target_label.to(args.device).long()
                label_batch_size = img_labeled.shape[0]
                
                noise = torch.clamp(torch.randn_like(img_unlabel) * 0.1, -0.2, 0.2)# 随机生成噪音
                ema_inputs = img_unlabel + noise
                ema_inputs = ema_inputs.to(args.device).float()

                volume_batch = torch.cat([img_labeled, img_unlabel], dim=0).to(args.device).float()

                outputs1 = model1(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = model2(volume_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)

                # supervised loss 两个监督loss
                loss1 = 0.5*(criterion(outputs1[:label_batch_size], target_label) +
                             dice_loss(outputs_soft1[:label_batch_size], target_label.unsqueeze(1)))

                loss2 = 0.5*(criterion(outputs2[:label_batch_size], target_label) +
                             dice_loss(outputs_soft2[:label_batch_size], target_label.unsqueeze(1)))

                loss_sup = loss1+loss2

                # 4个半监督损失=两个cps loss+两个mean_teacher loss
                # cross pseudo losses
                pseudo_outputs1 = torch.argmax(outputs_soft1[label_batch_size:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(outputs_soft2[label_batch_size:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = dice_loss(outputs_soft1[label_batch_size:], pseudo_outputs2.unsqueeze(1))
                pseudo_supervision2 = dice_loss(outputs_soft2[label_batch_size:], pseudo_outputs1.unsqueeze(1))

                # mean teacher losses
                consistency_weight_cps = args.consistency * linear_rampup(cur_itrs // 150, args.consistency_rampup)
                consistency_weight_mt = args.consistency * linear_rampup(cur_itrs // 150, args.consistency_rampup)

                if cur_itrs < 1000:
                    consistency_loss1 = 0.0
                    consistency_loss2 = 0.0
                else:
                    consistency_loss1 = torch.mean((outputs_soft1[label_batch_size:] - ema_output_soft) ** 2)
                    consistency_loss2 = torch.mean((outputs_soft2[label_batch_size:] - ema_output_soft) ** 2)

                model1_loss = 7 * consistency_weight_cps * pseudo_supervision1 + consistency_weight_mt * consistency_loss1
                model2_loss = 7 * consistency_weight_cps * pseudo_supervision2 + consistency_weight_mt * consistency_loss2

                loss_semi = model1_loss + model2_loss
                loss = loss_sup+loss_semi
                run_loss += loss.item()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                loss.backward()
                optimizer1.step()
                optimizer2.step()

                # ema方式进行更新
                update_ema_variables(model2, ema_model, args.ema_decay, cur_itrs)

                lr_scheduler1.step()
                lr_scheduler2.step()
                lr1 = optimizer1.param_groups[0]["lr"]
                lr2 = optimizer2.param_groups[0]["lr"]

                args.writer.add_scalar('S4CVnet/loss', loss.item(), cur_itrs)
                args.writer.add_scalar('S4CVnet/loss_semi', loss_semi.item(), cur_itrs)
                args.writer.add_scalar('S4CVnet/loss_sup', loss_sup.item(), cur_itrs)
                args.writer.add_scalar('S4CVnet/lr1', lr1, cur_itrs)
                args.writer.add_scalar('S4CVnet/lr2', lr2, cur_itrs)
                args.writer.add_scalar('S4CVnet/consistency_weight_cps', consistency_weight_cps, cur_itrs)
                args.writer.add_scalar('S4CVnet/consistency_weight_mt', consistency_weight_mt, cur_itrs)

                if cur_itrs % args.step_size == 0:
                    mean_dice, mean_hd952 = test_acdc(model=model1, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                    args.writer.add_scalar('S4CVnet/model1_dice', mean_dice, cur_itrs)
                    args.writer.add_scalar('S4CVnet/model1_hd95', mean_hd952, cur_itrs)

                    if mean_dice > best_dice1:
                        best_dice1 = mean_dice
                        torch.save({
                            "model": model1.state_dict(),
                            "optimizer": optimizer1.state_dict(),
                            "lr_scheduler": lr_scheduler1.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.model1_save_path)
                    model1.train()

                    #  模型2 进行测试
                    mean_dice, mean_hd952 = test_acdc(model=model2, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model2")
                    args.writer.add_scalar('S4CVnet/model2_dice', mean_dice, cur_itrs)
                    args.writer.add_scalar('S4CVnet/model2_hd95', mean_hd952, cur_itrs)

                    if mean_dice > best_dice2:
                        best_dice2 = mean_dice
                        torch.save({
                            "model": model2.state_dict(),
                            "optimizer": optimizer2.state_dict(),
                            "lr_scheduler": lr_scheduler2.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice2
                        }, args.model2_save_path)
                    model2.train()

                    mean_dice, mean_hd952 = test_acdc(model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                    args.writer.add_scalar('S4CVnet/model1_dice', mean_dice, cur_itrs)
                    args.writer.add_scalar('S4CVnet/model1_hd95', mean_hd952, cur_itrs)

                    if mean_dice > best_ema_dice:
                        best_ema_dice = mean_dice
                        torch.save({
                            "model": ema_model.state_dict(),
                            "optimizer": optimizer2.state_dict(),
                            "lr_scheduler": lr_scheduler2.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_ema_dice
                        }, args.ema_model_save_path)
                    ema_model.train()

                    args.logger.info("model1 best_dice: {:.4f} model2 best_dice: {:.4f} ema best_dice: {:.4f}".format(best_dice1, best_dice2,best_ema_dice))

                if cur_itrs > args.total_itrs:
                    return

                pbar.update(1)

            args.logger.info(
                "Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs, args.total_itrs,
                                                                 100. * cur_itrs / args.total_itrs,
                                                                 run_loss / len(unlabel_loader)))
    except Exception as e:
        args.logger.info(e)
        # print(e)
    finally:
        pbar.close()


if __name__ == "__main__":
    main()
