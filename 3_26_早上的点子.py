import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import random
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn.functional as F
from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss, Dense_Loss, BoxMaskGenerator
from model import build_model
from datasets import build_loader


def main():

    # path = r"config/ccnet_unet_30k_224x224_ACDC.yaml"
    path = r"config/ccnet_segformer_30k_224x224_ACDC.yaml"
    # path=r"config/ccnet_swinunet_30k_224x224_ACDC.yaml"
    # path=r"config/ccnet_cmt_30k_224x224_ACDC copy.yaml"
    # path = r"config/ccnet_uniformer_30k_224x224_ACDC.yaml"
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

    label_loader, unlabel_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = build_model(args=args).to(device=args.device)  # 创建模型
    ema_model = deepcopy(model)  # 创建ema_model
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # step 2: 训练模型
    CCNet(model, ema_model, label_loader, unlabel_loader, test_loader, args)


def CCNet(model, ema_model, label_loader, unlabel_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)
    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    med_loss = Med_Sup_Loss(args.num_classes)
    dense_loss = Dense_Loss(args.batch_size + args.unlabel_batch_size//2, args.device)

    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    model.train()
    ema_model.train()
    cur_itrs = 0

    best_dice1 = 0.0
    best_dice2 = 0.0

    class config:
        cutmix_mask_prop_range = (0.25, 0.5)
        cutmix_boxmask_n_boxes = 4
        cutmix_boxmask_fixed_aspect_ratio = False
        cutmix_boxmask_by_size = False
        cutmix_boxmask_outside_bounds = False
        cutmix_boxmask_no_invert = False

    mask_generator = BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                      n_boxes=config.cutmix_boxmask_n_boxes,
                                      random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                      prop_by_area=not config.cutmix_boxmask_by_size,
                                      within_bounds=not config.cutmix_boxmask_outside_bounds,
                                      invert=not config.cutmix_boxmask_no_invert)

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
                label_img, target_label = next(label_iter)

            label_bs = label_img.shape[0]
            unlabel_bs = unlabel_img.shape[0]
            label_img = label_img.to(args.device).float()
            unlabel_img = unlabel_img.to(args.device).float()
            cutmix_mask = mask_generator.generate_params(n_masks=unlabel_bs//2, mask_shape=(args.train_crop_size[0], args.train_crop_size[1]))
            cutmix_mask = torch.tensor(cutmix_mask, dtype=torch.float).to(args.device)
            target_label = target_label.to(args.device).long()

            unlabeled_volume_batch_0 = unlabel_img[:unlabel_bs // 2, ...]
            unlabeled_volume_batch_1 = unlabel_img[unlabel_bs // 2:, ...]

            batch_un_mix = unlabeled_volume_batch_0*(1.0-cutmix_mask)+unlabeled_volume_batch_1*cutmix_mask
            input_volume_batch = torch.cat([label_img, batch_un_mix], dim=0)

            output_mix, high_feature_mix, head_feature_mix = model(input_volume_batch)
            output_soft1 = torch.softmax(output_mix, dim=1)

            with torch.no_grad():
                ema_output_mix, ema_output_high_feature_mix, ema_output_head_feature_mix = ema_model(input_volume_batch)
                ema_output_ux0, ema_output_high_feature0, ema_output_head_feature0 = ema_model(unlabeled_volume_batch_0)
                ema_output_ux1, ema_output_high_feature1, ema_output_head_feature1 = ema_model(unlabeled_volume_batch_1)

                cutmix_mask = cutmix_mask.squeeze(1)

                conf_0, pseudo_outputs0 = torch.max(torch.softmax(ema_output_ux0.detach(), dim=1), dim=1)
                conf_1, pseudo_outputs1 = torch.max(torch.softmax(ema_output_ux1.detach(), dim=1), dim=1)
                conf_mixed = conf_0 * (1.0 - cutmix_mask) + conf_1 * cutmix_mask
                pseudo_pred_mixed = pseudo_outputs0 * (1.0 - cutmix_mask) + pseudo_outputs1 * cutmix_mask

                target_unlabel = pseudo_pred_mixed.detach()
                target_unlabel[conf_mixed < 0.7] = 255
                target_unlabel = target_unlabel.long()

            loss_sup_mix = med_loss(output_mix[:label_bs], target_label)
            loss_contrastive = dense_loss(high_feature_mix, ema_output_high_feature_mix)
            consistency_weight = get_current_consistency_weight(epoch=epoch, args=args)

            loss_u_ce = ce_loss(output_mix[label_bs:], target_unlabel)
            loss_u_dice = dice_loss(output_soft1[label_bs:], target_unlabel.unsqueeze(1))
            loss_consistence = 0.5*(loss_u_ce+loss_u_dice)

            loss = loss_sup_mix + consistency_weight * (loss_contrastive+loss_consistence)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            update_ema_variables(model, ema_model, args.ema_decay, cur_itrs)

            args.writer.add_scalar('mynet/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('mynet/lr', lr, cur_itrs)
            args.writer.add_scalar('mynet/consistency_weight', consistency_weight, cur_itrs)

            if cur_itrs % args.step_size == 0:
                mean_dice, mean_hd952 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model1")
                args.logger.info("model1 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mynet/model1_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mynet/model1_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice1:
                    best_dice1 = mean_dice
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_dice": best_dice1
                        }, os.path.join(args.save_path, "model", "model_{:.4f}.pth".format(best_dice1)))

                mean_dice, mean_hd952 = test_acdc(model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, name="model2")
                args.logger.info("model2 dice: {:.4f}, hd952: {:.4f}".format(mean_dice, mean_hd952))
                args.writer.add_scalar('mynet/model2_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('mynet/model2_hd95', mean_hd952, cur_itrs)

                if mean_dice > best_dice2:
                    best_dice2 = mean_dice
                    torch.save(
                        {
                            "model": ema_model,
                            "optimizer": optimizer,
                            "lr_scheduler": lr_scheduler,
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


def test_acdc(model, test_loader, args, cur_itrs, name="test"):
    """
    测试模型
    :param model: 模型
    :param test_loader:
    :param args:
    :param cur_itrs:
    :return:
    """
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(test_loader):
        image = sampled_batch[0].to(args.device)
        label = sampled_batch[1].to(args.device)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=args.test_crop_size)
        metric_list += np.array(metric_i)

        if i_batch == 0:
            slice = image[0, 0, :, :].cpu().detach().numpy()
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (args.test_crop_size[0] / x, args.test_crop_size[1] / y), order=0)
            img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(args.device)

            label_pred = torch.argmax(torch.softmax(model.val(img), dim=1), dim=1, keepdim=False).squeeze(0)
            label_pred = label_pred.cpu().detach().numpy()
            label_pred = zoom(label_pred, (x / args.test_crop_size[0], y / args.test_crop_size[1]), order=0)
            label_pred = test_loader.dataset.label_to_img(label_pred)

            label_true = label[0, 0, :, :].squeeze().cpu().detach().numpy()
            label_true = test_loader.dataset.label_to_img(label_true)

            args.writer.add_image('{}/Image'.format(name), img[0], cur_itrs)
            args.writer.add_image('{}/label_pred'.format(name), label_pred, cur_itrs, dataformats='HWC')
            args.writer.add_image('{}/label_true'.format(name), label_true, cur_itrs, dataformats='HWC')

    metric_list = metric_list / len(test_loader.dataset)
    performance2 = np.mean(metric_list, axis=0)[0]
    mean_hd952 = np.mean(metric_list, axis=0)[1]
    return performance2, mean_hd952


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net.val(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


if __name__ == '__main__':
    main()
