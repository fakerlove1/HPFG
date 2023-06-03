import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils import loadyaml, _get_logger, mk_path, build_lr_scheduler, build_optimizer, DiceLoss, get_current_consistency_weight
from utils import FeatureMemory, contrastive_class_to_class_learned_memory, VAT2d
from model import build_model
from datasets import build_loader
from val import test_acdc_ssnet


def main():

    path = r"config/ssnet_unet_30k_224x224_ACDC.yaml"

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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    label_loader, unlabel_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = build_model(args=args).to(device=args.device)  # 创建模型
    # ema_model = deepcopy(model)  # 创建ema_model
    ema_model = build_model(args=args).to(device=args.device)
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    prototype_memory = FeatureMemory(elements_per_class=32, n_classes=args.num_classes)

    # step 2: 训练模型
    SSNet(model, label_loader, unlabel_loader, test_loader, prototype_memory, args)


def SSNet(model, label_loader, unlabel_loader, test_loader, prototype_memory, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)

    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # config network and criterion
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)
    adv_loss = VAT2d(epi=args.magnitude)

    model.train()
    cur_itrs = 0
    train_loss = 0
    best_dice = 0.0

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

            volume_batch = torch.cat([img_labeled, img_unlabeled], dim=0)
            outputs, embedding = model(volume_batch)
            outputs_soft = F.softmax(outputs, dim=1)

            labeled_features = embedding[:label_bs, ...]
            unlabeled_features = embedding[label_bs:, ...]
            y = outputs_soft[:label_bs]
            _, prediction_label = torch.max(y, dim=1)
            _, pseudo_label = torch.max(outputs_soft[label_bs:], dim=1)  # Get pseudolabels
            mask_prediction_correctly = ((prediction_label == target_label).float() * (prediction_label > 0).float()).bool()

            # select the correct predictions and ignore the background class
            labeled_features = labeled_features.permute(0, 2, 3, 1)
            labels_correct = target_label[mask_prediction_correctly]  # [C] C是分类正确的像素点。全部分类正确为 c=b*h*w
            labeled_features_correct = labeled_features[mask_prediction_correctly, ...]  # [C,16]

            # get projected features
            with torch.no_grad():
                model.eval()
                proj_labeled_features_correct = model.projection_head(labeled_features_correct)  # [C,32]
                model.train()

            '''
            updated memory bank
            把分类正确的特征，放进memory bank中 进行存储
            memory_bank 是[num_classes,...] memory_bank[0] 表示第1类存储的特征
            '''
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)
            labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])  # [b*h*w,16]
            labeled_labels = target_label.reshape(-1)  # [b*h*w]

            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)  # [b*h*w,32]
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)  # [b*h*w,32]

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_class_to_class_learned_memory(model,
                                                                           pred_labeled_features_all,
                                                                           labeled_labels,
                                                                           args.num_classes,
                                                                           prototype_memory.memory)

            unlabeled_features = unlabeled_features.permute(0, 2, 3, 1).reshape(-1, labeled_features.size()[-1])
            pseudo_label = pseudo_label.reshape(-1)

            # get predicted features
            proj_feat_unlabeled = model.projection_head(unlabeled_features)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_class_to_class_learned_memory(model,
                                                                             pred_feat_unlabeled,
                                                                             pseudo_label,
                                                                             args.num_classes,
                                                                             prototype_memory.memory)

            loss_seg_ce = ce_loss(outputs[:label_bs], target_label[:].long())
            loss_seg_dice = dice_loss(y, target_label.unsqueeze(1))
            loss_lds = adv_loss(model, volume_batch)
            consistency_weight = get_current_consistency_weight(cur_itrs//150, args=args)
            loss = loss_seg_dice + consistency_weight * (loss_lds + 0.1 * (loss_contr_labeled + loss_contr_unlabeled))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            train_loss += loss.item()
            args.writer.add_scalar('SSNet/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('SSNet/lr', lr, cur_itrs)
            args.writer.add_scalar('SSNet/consistency_weight', consistency_weight, cur_itrs)
            args.writer.add_scalar('SSNet/loss_lds', loss_lds, cur_itrs)

            if cur_itrs % args.step_size == 0:
                dice, hd95 = test_acdc_ssnet(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="test_model")
                args.writer.add_scalar('SSNet/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('SSNet/{}_hd95'.format(args.name), hd95, cur_itrs)
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
                args.logger.info("epoch:{} \t ema_model dice:{:.4f} \t hd95:{:.4f} ".format(epoch, dice, hd95))
                args.logger.info("best model dice:{:.4f}".format(best_dice))
                model.train()

            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.4f} ".format(cur_itrs, args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss
                                                                          ))


if __name__ == '__main__':
    main()
