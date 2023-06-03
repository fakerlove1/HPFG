import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter


from train import ICT_MedSeg
from utils import loadyaml,_get_logger, mk_path
from model import get_model
from datasets import get_loader


if __name__ == '__main__':
    # paper link https://arxiv.org/abs/2202.00677
    path = r"config/2022-02-ISBI-ICT-MedSeg_Synapse.yaml"

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

    label_loader, unlabel_loader, test_loader = get_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # step 1: 构建模型
    model = get_model(args=args).to(device=args.device)  # 创建模型
    # ema_model = deepcopy(model)  # 创建ema_model
    ema_model = get_model(args=args).to(device=args.device)
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # step 2: 训练模型
    ICT_MedSeg(model=model, ema_model=ema_model, label_loader=label_loader, unlabl_loader=unlabel_loader,test_loader=test_loader, args=args)


