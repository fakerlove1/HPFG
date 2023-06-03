
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from skimage import io
from PIL import Image
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import DiceLoss, update_ema_variables, get_current_consistency_weight, Medical_Metric


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def hausdorff_95(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    return metric.binary.hd95(output, target)

# 灵敏度预测正确的样本占总阳性样本的比例,越大越好


def sensitivity_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / (target.sum() + smooth)

# 精确率表示预测为阳性的样本中,预测正确的比例,越大越好


def ppv_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / (output.sum() + smooth)


# def test_lidc(model, test_loader, args, cur_itrs, name="test"):
#     model.eval()
#     dice = 0.0
#     hd95 = 0.0
#     with torch.no_grad():
#         for i, (img, label_true) in enumerate(tqdm(test_loader)):
#             img = img.to(args.device).float()
#             label_true = label_true.to(args.device)
#             label_pred = model(img)  # [b,1,h,w]
#             dice += dice_coef(label_pred, label_true.unsqueeze(1)) * img.shape[0]
#             hd95 += hausdorff_95(label_pred, label_true.unsqueeze(1)) * img.shape[0]
#             if i == 0:
#                 img = make_grid(img, normalize=True, scale_each=True, nrow=8).permute(1, 2, 0).data.cpu().numpy()
#                 args.writer.add_image('{}/Image'.format(name), img, cur_itrs, dataformats='HWC')
#                 label_pred = torch.sigmoid(label_pred).squeeze(1).data.cpu().numpy()
#                 label_pred[label_pred > 0] = 1
#                 args.writer.add_image('{}/label_pred'.format(name), test_loader.dataset.label_to_img(label_pred), cur_itrs, dataformats='HWC')
#                 args.writer.add_image('{}/label_true'.format(name), test_loader.dataset.label_to_img(label_true), cur_itrs, dataformats='HWC')

#     dice /= len(test_loader.dataset)
#     hd95 /= len(test_loader.dataset)
#     return dice, hd95

def test_lidc(model, test_loader, args, cur_itrs, name="test"):
    model.eval()
    dice = 0.0
    hd95 = 0.0
    with torch.no_grad():
        for i, (img, label_true) in enumerate(tqdm(test_loader)):
            img = img.to(args.device).float()
            label_true = label_true.to(args.device).long()
            label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False)
            d, h = calculate_metric_percase(label_pred.data.cpu().numpy() == 1, label_true.data.cpu().numpy() == 1)
            dice += d*img.shape[0]
            hd95 += h*img.shape[0]
            if i == 0:
                img = make_grid(img, normalize=True, scale_each=True, nrow=8).permute(1, 2, 0).data.cpu().numpy()
                args.writer.add_image('{}/Image'.format(name), img, cur_itrs, dataformats='HWC')
                args.writer.add_image('{}/label_pred'.format(name), test_loader.dataset.label_to_img(label_pred), cur_itrs, dataformats='HWC')
                args.writer.add_image('{}/label_true'.format(name), test_loader.dataset.label_to_img(label_true), cur_itrs, dataformats='HWC')

    dice /= len(test_loader.dataset)
    hd95 /= len(test_loader.dataset)
    return dice, hd95


def test_isic(model, test_loader, args, cur_itrs, name="test"):
    model.eval()
    dice = 0.0
    hd95 = 0.0
    with torch.no_grad():
        for i, (img, label_true) in enumerate(tqdm(test_loader)):
            img = img.to(args.device).float()
            label_true = label_true.to(args.device).long()
            label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False)
            d, h = calculate_metric_percase(label_pred.data.cpu().numpy() == 1, label_true.data.cpu().numpy() == 1)
            dice += d*img.shape[0]
            hd95 += h*img.shape[0]
            if i == 0: 
                img = make_grid(img, normalize=True, scale_each=True, nrow=8).permute(1, 2, 0).data.cpu().numpy()
                args.writer.add_image('{}/Image'.format(name), img, cur_itrs, dataformats='HWC')
                args.writer.add_image('{}/label_pred'.format(name), test_loader.dataset.label_to_img(label_pred), cur_itrs, dataformats='HWC')
                args.writer.add_image('{}/label_true'.format(name), test_loader.dataset.label_to_img(label_true), cur_itrs, dataformats='HWC')

    dice /= len(test_loader.dataset)
    hd95 /= len(test_loader.dataset)
    return dice, hd95


def test_acdc(model, test_loader, args, cur_itrs, name="test"):
    """
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
            img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False).squeeze(0)
            label_pred = label_pred.cpu().detach().numpy()
            label_pred = zoom(label_pred, (x / args.test_crop_size[0], y / args.test_crop_size[1]), order=0)
            label_pred = test_loader.dataset.label_to_img(label_pred)

            label_true = label[0, 0, :, :].squeeze().cpu().detach().numpy()
            label_true = test_loader.dataset.label_to_img(label_true)

            args.writer.add_image('{}/Image'.format(name), img[0], cur_itrs)
            args.writer.add_image('{}/label_pred'.format(name), label_pred, cur_itrs, dataformats='HWC')
            args.writer.add_image('{}/label_true'.format(name), label_true, cur_itrs, dataformats='HWC')

    metric_list = metric_list / len(test_loader.dataset)

    args.logger.info("class dice:{}".format(metric_list[:,0]))
    # args.logger.info("class dice:{}".format(len(metric_list[:,0])))
    performance2 = np.mean(metric_list, axis=0)[0]
    mean_hd952 = np.mean(metric_list, axis=0)[1]
    return performance2, mean_hd952


def test_synapse(model, test_loader, args, cur_itrs, name="test"):
    """
    :param model: 模型
    :param test_loader:
    :param args:
    :param cur_itrs:
    :return:
    """
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(tqdm(test_loader)):
        image = sampled_batch[0].to(args.device)
        label = sampled_batch[1].to(args.device)
        metric_i = test_single_volume_synapse(image, label, model, classes=args.num_classes, patch_size=args.test_crop_size)
        metric_list += np.array(metric_i)

        if i_batch == 0:
            slice = image[0, 0, :, :].cpu().detach().numpy()
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (args.test_crop_size[0] / x, args.test_crop_size[1] / y), order=0)
            img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False).squeeze(0)
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


def test_single_volume_synapse(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


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
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    # print(metric_list)
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_acdc_ssnet(model, test_loader, args, cur_itrs, name="test"):
    """
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
        metric_i = test_single_volume_ssnet(image, label, model, classes=args.num_classes, patch_size=args.test_crop_size)
        metric_list += np.array(metric_i)

        if i_batch == 0:
            slice = image[0, 0, :, :].cpu().detach().numpy()
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (args.test_crop_size[0] / x, args.test_crop_size[1] / y), order=0)
            img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            out = model(img)
            if len(out) > 1:
                out = out[0]
            label_pred = torch.argmax(torch.softmax(out, dim=1), dim=1, keepdim=False).squeeze(0)
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


def test_single_volume_ssnet(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = net(input)
            if len(out) > 1:
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
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

    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def visual(args, model, img, label_pred, label_true, cur_itrs, loader):
    """
    特征图可视化
    :param args:
    :param model: 模型
    :param img: 图片
    :param label_pred: 预测结果
    :param label_true: 真实结果
    :param cur_itrs: 当前iter
    :param loader: 数据集
    :return:
    """
    model.eval()

    #  添加卷积层参数
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     # print(name)
    #     if "backbone.conv1" in name or "backbone.layer4.2.conv3" in name or "cat_conv.5.weight" in name:
    #         args.writer.add_histogram(name, param, cur_itrs)
    # 获取特征图

    # 记录图片
    img_grid = make_grid(tensor=img.data.cpu(), nrow=2,
                         normalize=True, scale_each=True, )
    args.writer.add_image("train/img", img_grid, cur_itrs)

    label_pred = label_pred.detach().max(dim=1)[1].data.cpu().numpy()
    args.writer.add_image("train/label_pred",
                          loader.dataset.label_to_img(label_pred), cur_itrs, dataformats='HWC')

    label_true = label_true.data.cpu().numpy()
    args.writer.add_image("train/label_true",
                          loader.dataset.label_to_img(label_true), cur_itrs, dataformats='HWC')

    if isinstance(model, nn.DataParallel):
        with torch.no_grad():
            features = model.module.backbone(img)
    else:
        with torch.no_grad():
            features = model.backbone(img)

    if len(features) == 4:
        [_, low_level_features, _, x] = features
    elif len(features) == 2:
        low_level_features, x = features
    else:
        raise ("visual error")
    #  特征图
    low_level_features = make_grid(low_level_features[0].detach().cpu().unsqueeze(dim=1),
                                   normalize=True,
                                   scale_each=True,
                                   nrow=8)  # normalize进行归一化处理
    args.writer.add_image("backbone/low_level_features",
                          low_level_features, cur_itrs)
    x = make_grid(x[0].detach().cpu().unsqueeze(dim=1),
                  normalize=True,
                  scale_each=True,
                  nrow=8)  # normalize进行归一化处理
    args.writer.add_image("backbone/high_level_features", x, cur_itrs)

    # for name, layer in model._modules.items():
    #
    #     x = layer(x)
    #
    #     if 'backbone' in name or 'conv' in name:
    #         x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
    #         img_grid = make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
    #         writer.add_image(f'{name}_feature_maps', img_grid, global_step=cur_itrs)
