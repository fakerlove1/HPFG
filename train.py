import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from utils import get_lr_scheduler, get_optimizer, linear_rampup,Med_Sup_Loss,DiceLoss, update_ema_variables, get_current_consistency_weight
from val import test_acdc,test_lidc
from utils import BCEDiceLoss




def Supervise(model, train_loader, test_loader, args):
    optimizer = get_optimizer(args=args, model=model)

    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)

    max_epoch = args.total_itrs // len(train_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    model.train()
    cur_itrs = 0
    train_loss = 0.0
    best_dice = 0.0

    #  加载原模型
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        cur_itrs = state_dict["cur_itrs"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        # lr_scheduler = state_dict["scheduler"]
        best_dice = state_dict["best_score"]

    for epoch in range(max_epoch):
        for i, (img_labeled, target_label) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
            img_labeled = img_labeled.to(args.device).float()
            target_label = target_label.to(args.device).long()
            pseudo_labeled = model(img_labeled)
            loss_ce = criterion(pseudo_labeled, target_label)
            loss_dice = dice_loss(pseudo_labeled, target_label.unsqueeze(1), softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            train_loss += loss.item()
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)

            if cur_itrs % args.step_size == 0:
                dice, hd95 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs)
                args.writer.add_scalar('supervise/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('supervise/{}_hd95'.format(args.name), hd95, cur_itrs)
                args.logger.info("epoch:{} \t dice:{:.5f} \t hd95:{:.5f} ".format(epoch, dice, hd95))

                if dice > best_dice:
                    best_dice = dice
                    torch.save({
                        "cur_itrs":cur_itrs,
                        "best_dice":best_dice,
                        "model":model,
                        "optimizer":optimizer,
                        "lr_scheduler":lr_scheduler,
                    },args.model_save_path)

                model.train()

            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f} ".format(cur_itrs, args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss
                                                                          ))
        train_loss = 0


def CLLE_MegSeg():
    """
    
    """
    pass




def S4CVnet(model1, model2, ema_model, label_loader, unlabl_loader, test_loader, args):
    
    optimizer1 = get_optimizer(args=args.model1, model=model1)
    lr_scheduler1 = get_lr_scheduler(args=args.model1, optimizer=optimizer1)

    optimizer2 = get_optimizer(args=args.model2, model=model2)
    lr_scheduler2 = get_lr_scheduler(args=args.model2, optimizer=optimizer2)

    max_epoch = args.total_itrs // len(unlabl_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    model1.train()
    model2.train()
    cur_itrs = 0
    best_dice1 = 0.0
    best_dice2 = 0.0
    label_iter = iter(label_loader)

    try:
        if args.process:
            pbar = tqdm(total=args.total_itrs)
        else:
            pbar = tqdm(total=args.total_itrs, file=args.tqdm)

        for epoch in range(max_epoch):
            run_loss = 0.0
            for idx, (img_unlabel, _) in enumerate(unlabl_loader):
                cur_itrs += 1
                try:
                    img_labeled, target_label = next(label_iter)
                except StopIteration:
                    label_iter = iter(label_loader)
                    img_labeled, target_label, = next(label_iter)

                target_label = target_label.to(args.device).long()
                label_batch_size = img_labeled.shape[0]
                # 随机生成噪音
                noise = torch.clamp(torch.randn_like(
                    img_unlabel) * 0.1, -0.2, 0.2)
                ema_inputs = img_unlabel + noise
                ema_inputs = ema_inputs.to(args.device).float()

                volume_batch = torch.cat(
                    [img_labeled, img_unlabel], dim=0).to(args.device).float()

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
                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[label_batch_size:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[label_batch_size:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = dice_loss(
                    outputs_soft1[label_batch_size:], pseudo_outputs2.unsqueeze(1))
                pseudo_supervision2 = dice_loss(
                    outputs_soft2[label_batch_size:], pseudo_outputs1.unsqueeze(1))

                # mean teacher losses
                consistency_weight_cps = args.consistency * \
                    linear_rampup(cur_itrs // 150, args.consistency_rampup)
                consistency_weight_mt = args.consistency * \
                    linear_rampup(cur_itrs // 150, args.consistency_rampup)

                if cur_itrs < 1000:
                    consistency_loss1 = 0.0
                    consistency_loss2 = 0.0
                else:
                    consistency_loss1 = torch.mean(
                        (outputs_soft1[label_batch_size:] - ema_output_soft) ** 2)
                    consistency_loss2 = torch.mean(
                        (outputs_soft2[label_batch_size:] - ema_output_soft) ** 2)

                model1_loss = 7 * consistency_weight_cps * pseudo_supervision1 + \
                    consistency_weight_mt * consistency_loss1
                model2_loss = 7 * consistency_weight_cps * pseudo_supervision2 + \
                    consistency_weight_mt * consistency_loss2

                loss_semi = model1_loss + model2_loss
                loss = loss_sup+loss_semi
                run_loss += loss.item()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                loss.backward()
                optimizer1.step()
                optimizer2.step()

                # ema方式进行更新
                model2, ema_model = update_ema_variables(
                    model2, ema_model, args.ema_decay, cur_itrs)

                lr_scheduler1.step()
                lr_scheduler2.step()
                lr1 = optimizer1.param_groups[0]["lr"]
                lr2 = optimizer2.param_groups[0]["lr"]

                args.writer.add_scalar('S4CVnet/loss', loss.item(), cur_itrs)
                args.writer.add_scalar(
                    'S4CVnet/loss_semi', loss_semi.item(), cur_itrs)
                args.writer.add_scalar(
                    'S4CVnet/loss_sup', loss_sup.item(), cur_itrs)
                args.writer.add_scalar('S4CVnet/lr1', lr1, cur_itrs)
                args.writer.add_scalar('S4CVnet/lr2', lr2, cur_itrs)
                args.writer.add_scalar(
                    'S4CVnet/consistency_weight_cps', consistency_weight_cps, cur_itrs)
                args.writer.add_scalar(
                    'S4CVnet/consistency_weight_mt', consistency_weight_mt, cur_itrs)

                if cur_itrs % args.step_size == 0:
                    # 可视化train 结果，包括backbone部分的数据，以及label
                    # visual(args=args, model=model, img=img, label_pred=out,
                    #        label_true=label, cur_itrs=cur_itrs, loader=train_loader)
                    # 开始测试
                    mean_dice, mean_hd952 = test_acdc(model=model1, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                    args.writer.add_scalar('S4CVnet/model1_dice', mean_dice, cur_itrs)
                    args.writer.add_scalar('S4CVnet/model1_hd95', mean_hd952, cur_itrs)

                    if mean_dice > best_dice1:
                        best_dice1 = mean_dice
                        torch.save({
                            "model": model1,
                            "optimizer": optimizer1,
                            "lr_scheduler": lr_scheduler1,
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, args.model1_save_path)
                    model1.train()

                    #  模型2 进行测试
                    mean_dice, mean_hd952 = test_acdc(
                        model=model2, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model2")
                    args.writer.add_scalar(
                        'S4CVnet/model2_dice', mean_dice, cur_itrs)
                    args.writer.add_scalar(
                        'S4CVnet/model2_hd95', mean_hd952, cur_itrs)

                    if mean_dice > best_dice2:
                        best_dice2 = mean_dice
                        torch.save({
                            "model": model2,
                            "optimizer": optimizer2,
                            "lr_scheduler": lr_scheduler2,
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice2
                        }, args.model2_save_path)
                    model2.train()

                    args.logger.info("model1 best_dice: {:.5f} model2 best_dice: {:.5f}".format(
                        best_dice1, best_dice2))

                if cur_itrs > args.total_itrs:
                    return

                pbar.update(1)

            args.logger.info(
                "Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs, args.total_itrs,
                                                                 100. * cur_itrs / args.total_itrs,
                                                                 run_loss / len(unlabl_loader)))
    except Exception as e:
        args.logger.info(e)
        # print(e)
    finally:
        pbar.close()


