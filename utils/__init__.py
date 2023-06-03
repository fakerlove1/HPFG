from .knn_monitor import knn_monitor, knn_predict,_topk_retrieval
from .logger import _get_logger
from .loss import FocalLoss, DiceLoss,Med_Sup_Loss,SimCLRLoss,SimSiamLoss,ContrastiveLoss,BCEDiceLoss,BCEDiceL1Loss
from .loss import FeatureMemory,contrastive_class_to_class_learned_memory,VAT2d,Pixel_Class_Contrastive_Loss,Dense_Loss
from .utils import *
from .scheduler import *
from .metric import SegMetrics, Medical_Metric, AverageMeter
import torch




def build_optimizer(args, model):
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt == "adamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ("get_optimizer error")

    return optimizer

def build_lr_scheduler(args, optimizer):
    if args.sched == "cosine":
        lr_scheduler = CosineWarmupLR_Scheduler(
                    optimizer=optimizer,
                    base_lr=args.lr,
                    warmup_epochs=args.warmup_epochs, 
                    warmup_lr=args.warmup_lr,
                    final_lr=args.min_lr,
                    iter_per_epoch=args.step_size,
                    num_epochs=args.total_itrs//args.step_size)
        
    elif args.sched == "poly":
        lr_scheduler = PolyLR(optimizer, max_iters=args.total_itrs,power=0.1,min_lr=args.min_lr)
    elif args.sched=="medical":
        lr_scheduler = Medical_LR(optimizer=optimizer,base_lr=args.lr,max_iterations=args.total_itrs)
    elif args.sched == "":
        lr_scheduler = create_lr_scheduler(optimizer, num_step=args.step_size, epochs=args.epochs,
                                           warmup=True, warmup_epochs=3, warmup_factor=1e-4)

    else:
        raise ("get_lr_scheduler error")
    return lr_scheduler

