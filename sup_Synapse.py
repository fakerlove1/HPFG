import os.path
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm


from utils import loadyaml, _get_logger, mk_path,DiceLoss
from model import build_model
from datasets import build_loader
from utils import build_lr_scheduler,build_optimizer
from val import test_synapse

def main():

    path = r"config/unet_30k_224x224_ISIC.yaml"
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



def Supervise(model, train_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)

    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)

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
        lr_scheduler = state_dict["lr_scheduler"]
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
                dice, hd95 = test_synapse(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs)
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
                    },args.supervise_save_path)

                model.train()

            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f} ".format(cur_itrs, args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss
                                                                          ))
        train_loss = 0



if __name__ == '__main__':
    main()