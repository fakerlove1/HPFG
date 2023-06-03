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
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss, Dense_Loss, BoxMaskGenerator,SegMetrics
from model import build_model
from datasets import build_loader


def main():

    # path = r"config/ccnet_unet_80k_100%_512x512_Building.yaml"
    path=r"config/ccnet_segformer_80k_100%_512x512_Building.yaml"
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

    train_loader, val_loader,test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(train_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(val_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = build_model(args=args).to(device=args.device)  # 创建模型
    ema_model = deepcopy(model)  # 创建ema_model
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # step 2: 训练模型
    CCNet(model, ema_model, train_loader, val_loader, args)


def CCNet(model, ema_model, train_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)
    max_epoch = args.total_itrs // len(train_loader) + 1
    # med_loss = Med_Sup_Loss(args.num_classes)
    med_loss = Med_Sup_Loss(args.num_classes,ce=0.4,dice=0.6)
    dense_loss = Dense_Loss(args.batch_size*2, args.device)

    metrics = SegMetrics(args.num_classes)  # 定义评价指标

    model.train()
    ema_model.train()
    cur_itrs = 0

    best_miou1 = 0.0
    best_miou2 = 0.0

    args.logger.info("max epoch: {}".format(max_epoch))
    args.logger.info("start training")
    pbar = tqdm(total=args.total_itrs)
    
    label_iter = iter(train_loader)
    for epoch in range(max_epoch):
        train_loss = 0.0
        for unlabel_img, _ in train_loader:
            cur_itrs += 1
            try:
                label_img, target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(train_loader)
                label_img, target_label = next(label_iter)

            label_bs = label_img.shape[0]
            unlabel_bs = unlabel_img.shape[0]
            label_img = label_img.to(args.device).float()
            unlabel_img = unlabel_img.to(args.device).float()
            target_label = target_label.to(args.device).long()

            input_volume_batch = torch.cat([label_img, unlabel_img], dim=0)

            output_mix, high_feature_mix, head_feature_mix = model(input_volume_batch)
            output_soft = torch.softmax(output_mix, dim=1)

            with torch.no_grad():
                ema_output_mix, ema_output_high_feature_mix, ema_output_head_feature_mix = ema_model(input_volume_batch)

            loss_sup_mix = med_loss(output_mix[:label_bs], target_label)
            loss_contrastive = dense_loss(high_feature_mix, ema_output_high_feature_mix)
            loss_consistence = torch.mean((output_soft[label_bs:] - ema_output_mix[label_bs:]) ** 2)
            consistency_weight = get_current_consistency_weight(epoch=cur_itrs // 150, args=args)
            loss = loss_sup_mix + consistency_weight * (loss_consistence+loss_contrastive)
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
                result = test(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, 
                                             metrics=metrics,name="model1")
                mean_miou=result["Mean IoU"]
                mean_acc=result["Mean Acc"]
                args.logger.info("model1 miou: {:.4f}, acc: {:.4f}".format(mean_miou, mean_acc))
                args.writer.add_scalar('mynet/model1_mean_miou', mean_miou, cur_itrs)
                args.writer.add_scalar('mynet/model1_mean_acc', mean_acc, cur_itrs)

                if mean_miou > best_miou1:
                    best_miou1 = mean_miou
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_miou": best_miou1
                        }, args.model_save_path)

                result = test(model=ema_model, test_loader=test_loader, args=args, cur_itrs=cur_itrs, 
                                             metrics=metrics,name="model2")
                mean_miou=result["Mean IoU"]
                mean_acc=result["Mean Acc"]
                args.logger.info("model2 miou: {:.4f}, acc: {:.4f}".format(mean_miou, mean_acc))
                args.writer.add_scalar('mynet/model2_mean_miou', mean_miou, cur_itrs)
                args.writer.add_scalar('mynet/model2_mean_acc', mean_acc, cur_itrs)

                if mean_miou > best_miou2:
                    best_miou2 = best_miou1
                    torch.save(
                        {
                            "model": ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "cur_itrs": cur_itrs,
                            "best_miou": best_miou2
                        }, args.ema_model_save_path)

                args.logger.info("model1 best_miou: {:.4f}, model2 best_miou: {:.4f}".format(best_miou1, best_miou2))
                model.train()
                ema_model.train()

            if cur_itrs > args.total_itrs:
                return
            
            pbar.update(1)

        args.logger.info("Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs,
                                                                          args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss / len(train_loader)))


def test(cur_itrs, model, test_loader, metrics, args,name="test"):
    model.eval()
    metrics.reset()
    args.logger.info("=========> test cur_itrs:{}".format(cur_itrs))
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):

            images = images.to(args.device).float()
            labels = labels.to(args.device).long()

            out = model.val(images)
            label_pred = out.detach().max(dim=1)[1].data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            metrics.update(label_true, label_pred)

            # 展示图
            if idx == 0:
                img_grid = make_grid(tensor=images.data.cpu(), nrow=2, normalize=True, scale_each=True)
                args.writer.add_image('{}/Image'.format(name), img_grid, cur_itrs)
                args.writer.add_image('{}/label_pred'.format(name), test_loader.dataset.label_to_img(label_pred),cur_itrs,dataformats='HWC')
                args.writer.add_image('{}/label_true'.format(name), test_loader.dataset.label_to_img(label_true),cur_itrs,dataformats='HWC')

    score = metrics.get_results()

    return score


# def test_acdc(model, test_loader, args, cur_itrs, name="test"):
#     """
#     测试模型
#     :param model: 模型
#     :param test_loader:
#     :param args:
#     :param cur_itrs:
#     :return:
#     """
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in enumerate(test_loader):
#         image = sampled_batch[0].to(args.device)
#         label = sampled_batch[1].to(args.device)
#         metric_i = test_single_volume(
#             image, label, model, classes=args.num_classes, patch_size=args.test_crop_size)
#         metric_list += np.array(metric_i)

#         if i_batch == 0:
#             slice = image[0, 0, :, :].cpu().detach().numpy()
#             x, y = slice.shape[0], slice.shape[1]
#             slice = zoom(
#                 slice, (args.test_crop_size[0] / x, args.test_crop_size[1] / y), order=0)
#             img = torch.from_numpy(slice).unsqueeze(
#                 0).unsqueeze(0).float().to(args.device)

#             label_pred = torch.argmax(torch.softmax(
#                 model.val(img), dim=1), dim=1, keepdim=False).squeeze(0)
#             label_pred = label_pred.cpu().detach().numpy()
#             label_pred = zoom(
#                 label_pred, (x / args.test_crop_size[0], y / args.test_crop_size[1]), order=0)
#             label_pred = test_loader.dataset.label_to_img(label_pred)

#             label_true = label[0, 0, :, :].squeeze().cpu().detach().numpy()
#             label_true = test_loader.dataset.label_to_img(label_true)

#             args.writer.add_image('{}/Image'.format(name), img[0], cur_itrs)
#             args.writer.add_image(
#                 '{}/label_pred'.format(name), label_pred, cur_itrs, dataformats='HWC')
#             args.writer.add_image(
#                 '{}/label_true'.format(name), label_true, cur_itrs, dataformats='HWC')

#     metric_list = metric_list / len(test_loader.dataset)
#     performance2 = np.mean(metric_list, axis=0)[0]
#     mean_hd952 = np.mean(metric_list, axis=0)[1]
#     return performance2, mean_hd952


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
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
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
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
