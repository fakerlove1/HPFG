from medpy import metric
import numpy as np


class Medical_Metric():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.class_dice = 0.0
        self.class_hd95 = 0.0
        self.n = 0

    def update(self, label_preds, label_trues):
        """
        2d的
        :param label_trues:
        :param label_preds:
        :return:
        """
        #  记录每个类的dice,
        for label_pred, label_true in zip(label_preds, label_trues):
            #  记录每个类的dice,
            class_dice = []
            class_hd95 = []
            for i in range(1, self.n_classes):
                result = self.calculate_metric_percase(label_pred == i, label_true == i)
                class_dice.append(result[0])
                class_hd95.append(result[1])
            self.class_dice += np.array(class_dice)
            self.class_hd95 += np.array(class_hd95)
            self.n = self.n + 1

    def get_results(self):
        class_hd95 = self.class_hd95 / self.n
        class_dice = self.class_dice / self.n
        mean_dice1 = np.mean(class_dice)
        mean_hd95 = np.mean(class_hd95)

        print("n:", self.n)
        print("mean_dice : {:.5f}".format(mean_dice1))
        print("hd952 : {:.5f}".format(mean_hd95))
        for i, item in enumerate(class_dice):
            print("class {}: {:.5f}".format(i, item))
        return {
            "mean_dice": mean_dice1,
            "mean_hd95": mean_hd95,
            "class_dice": class_dice,
        }

    def reset(self):
        self.class_dice = 0.0
        self.class_hdp5 = 0.0
        self.n = 0

    def to_str(self, metrics):
        print("mean_dice: {:.5f}\t mean_hd95: {:.5f}".format(metrics["mean_dice"], metrics["mean_hd95"]))
        for i, item in enumerate(metrics["class_dice"]):
            print("class:{} \t dice: {:.5f}".format(i + 1, item))

    @staticmethod
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95
        elif pred.sum() > 0 and gt.sum() == 0:
            return 1, 0
        else:
            return 0, 0


import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class SegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results, classes):
        string = "\n"
        # for k, v in results.items():
        #     if k != "Class IoU":
        #         string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "|class {:>15}: {:.5f}|\n".format(classes[k], v)
        string += "|Mean IoU: {:.5f}|\n".format(results["Mean IoU"])
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
