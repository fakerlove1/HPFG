from .swinunet import get_swinunet, get_swinunet_plus
from .unet import UNet, UNet_Plus
from .unet_large import UNet_Large
from .transunet import get_transunet
from .unet_LIDC import UNet_LIDC
from .swinunet_LIDC import get_swinunet_LIDC
from .transunet_LIDC import get_transunet_LIDC
from .segformer import SegFormer, SegFormer_Plus
from .ss_net import SSNet
from .swin_mae import swin_mae
from .cmt import CMT_Plus,CMT_S
from .uniformer import Uniformer_Plus

def build_model(args):
    if args.model == 'swinunet':
        if isinstance(args.train_crop_size, list) or isinstance(args.train_crop_size, tuple):
            image_size = args.train_crop_size[0]
        else:
            image_size = args.train_crop_size
        model = get_swinunet(img_size=image_size, num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "swinunet_plus":
        if isinstance(args.train_crop_size, list) or isinstance(args.train_crop_size, tuple):
            image_size = args.train_crop_size[0]
        else:
            image_size = args.train_crop_size
        model = get_swinunet_plus(img_size=image_size, num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "ssnet":
        model = SSNet(in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == "unet_plus":
        model = UNet_Plus(in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'transunet':
        model = get_transunet(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'transunet_lidc':
        model = get_transunet_LIDC(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'unet_large':
        model = UNet_Large(in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'unet_lidc':
        model = UNet_LIDC(in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == 'swinunet_lidc':
        if isinstance(args.train_crop_size, list) or isinstance(args.train_crop_size, tuple):
            image_size = args.train_crop_size[0]
        else:
            image_size = args.train_crop_size
        model = get_swinunet_LIDC(img_size=image_size, num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == 'segformer':
        model = SegFormer(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model == "segformer_plus":
        model = SegFormer_Plus(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model=="swinmae":
        model=swin_mae(in_channels=args.in_channels,mask_ratio=args.mask_ratio)
    elif args.model=="cmt":
        model = CMT_S(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model=="cmt_plus":
        model = CMT_Plus(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    elif args.model=="uniformer_plus":
        model=Uniformer_Plus(image_size=args.train_crop_size, in_channels=args.in_channels, num_classes=args.num_classes)
    else:
        raise NotImplementedError

    return model


def build_backbone(args):
    pass
