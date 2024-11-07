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