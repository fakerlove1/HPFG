from .ACDC import get_acdc_loader, get_ssl_acdc_loader
from .LIDC import get_lidc_loader, get_ssl_lidc_loader
from .Synapse import get_synapse_loader, get_ssl_synapse_loader
from .ISIC import get_isic_loader, get_ssl_isic_loader
from .Building import get_building_loader



def build_loader(args):
    if args.datasets == "acdc":
        label_loader, unlabel_loader, test_loader = get_ssl_acdc_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size,
            unlabel_batch_size=args.unlabel_batch_size,
            label_num=args.label_num)
        return label_loader, unlabel_loader, test_loader
    elif args.datasets == "lidc":
        label_loader, unlabel_loader, test_loader = get_ssl_lidc_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size,
            unlabel_batch_size=args.unlabel_batch_size,
            label_num=args.label_num)
        return label_loader, unlabel_loader, test_loader

    elif args.datasets == "synapse":
        label_loader, unlabel_loader, test_loader = get_ssl_synapse_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size,
            unlabel_batch_size=args.unlabel_batch_size,
            label_num=args.label_num)
        return label_loader, unlabel_loader, test_loader
    elif args.datasets == "isic":
        label_loader, unlabel_loader, test_loader = get_ssl_isic_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            unlabel_batch_size=args.unlabel_batch_size,
            train_crop_size=args.train_crop_size,
            label_num=args.label_num)
        return label_loader, unlabel_loader, test_loader
    
    elif args.datasets == "sup_lidc":
        train_loader, test_loader = get_lidc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader

    elif args.datasets == "sup_acdc":
        train_loader, test_loader = get_acdc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader

    elif args.datasets == "sup_synapse":
        train_loader, test_loader = get_synapse_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader
    elif args.datasets == "sup_isic":
        train_loader, test_loader = get_isic_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader
    elif args.datasets == "sup_building":
        train_loader, val_loader,test_loader = get_building_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, val_loader,test_loader
    else:
        raise NotImplementedError
