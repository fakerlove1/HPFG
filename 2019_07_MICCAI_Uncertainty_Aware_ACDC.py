import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import random
import torch.nn.functional as F
from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables
from utils import build_lr_scheduler, build_optimizer, sigmoid_rampup
from model import build_model
from datasets import build_loader
from val import test_acdc

def main():
    path = r"config/uncertainty_aware_unet_30k_224x224_ACDC.yaml"
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
    Uncertainty_Aware(model, ema_model, label_loader, unlabel_loader, test_loader, args)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def Uncertainty_Aware(model, ema_model, label_loader, unlabel_loader, test_loader, args):
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
        for i, (unlabeled_volume_batch, _) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1
            try:
                img_labeled, target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                img_labeled, target_label, = next(label_iter)

            img_labeled = img_labeled.to(args.device).float()
            target_label = target_label.to(args.device).long()
            unlabeled_volume_batch = unlabeled_volume_batch.to(args.device).float()
            label_bs = img_labeled.shape[0]
            volume_batch = torch.cat([img_labeled, unlabeled_volume_batch], dim=0)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                ema_output = ema_model(ema_inputs)

            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, args.num_classes, w, h]).cuda()

            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)

            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, args.num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)

            loss_ce = criterion(outputs[:label_bs], target_label)
            loss_dice = dice_loss(outputs_soft[:label_bs], target_label.unsqueeze(1))

            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(epoch=cur_itrs // 150, args=args)
            consistency_dist = softmax_mse_loss(outputs[label_bs:], ema_output)  # (batch, 2, 112,112,80)

            threshold = (0.75 + 0.25 * sigmoid_rampup(cur_itrs, args.total_itrs))*np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist)/(2 * torch.sum(mask) + 1e-16)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            update_ema_variables(model, ema_model, args.ema_decay, cur_itrs)

            train_loss += loss.item()
            args.writer.add_scalar('Uncertainty_Aware/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('Uncertainty_Aware/lr', lr, cur_itrs)
            args.writer.add_scalar('Uncertainty_Aware/consistency_weight', consistency_weight, cur_itrs)
            args.writer.add_scalar('Uncertainty_Aware/consistency_loss', consistency_loss, cur_itrs)
            args.writer.add_scalar('Uncertainty_Aware/threshold', threshold, cur_itrs)


            if cur_itrs % args.step_size == 0:
                dice, hd95 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="test_model")
                args.writer.add_scalar('Uncertainty_Aware/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('Uncertainty_Aware/{}_hd95'.format(args.name), hd95, cur_itrs)
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

                dice, hd95 = test_acdc(
                    model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="test_ema_model")

                if dice > best_ema_dice:
                    best_ema_dice = dice
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice,
                        "model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, args.ema_model_save_path)

                args.writer.add_scalar('Uncertainty_Aware/{}_dice_ema'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('Uncertainty_Aware/{}_hd95_ema'.format(args.name), hd95, cur_itrs)
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
