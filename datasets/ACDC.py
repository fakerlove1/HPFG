import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import h5py
from torchvision.utils import make_grid
from .utils import RandomGenerator,TwoStreamBatchSampler,patients_to_slices


class ACDC(Dataset):

    PALETTE = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ])

    def __init__(self, root=r"E:\note\ssl\data\ACDC", split="train", transform=None):

        super(ACDC, self).__init__()
        self.split = split
        self.root = root
        self.transform = transform
        self.sample_list = []
        self.load_annotations()  # 加载文件路径
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, "r")

        image = np.array(h5f["image"][:],dtype=np.float32)
        mask = np.array(h5f["label"][:],dtype=np.uint8)

        if self.transform is not None and self.split == "train":
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        return image, mask

    def label_to_img(self, label):
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=2, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

    def load_annotations(self):
        if self.split == "train":
            with open(self.root + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/slices/{}.h5".format(item) for item in self.sample_list]
        elif self.split == "val":
            with open(self.root + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/{}.h5".format(item) for item in self.sample_list]
        else:
            with open(self.root + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/{}.h5".format(item) for item in self.sample_list]
            
        self.sample_list=np.array(self.sample_list)


def get_acdc_loader(root=r'/home/ubuntu/data/ACDC', batch_size=4, train_crop_size=(224, 224)):
    """

    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1],scale=(0.5, 2.0)),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.8),
        A.ColorJitter(0.4,0.4,0.4,p=0.5),
        ToTensorV2()
    ])
    train_transform=RandomGenerator(train_crop_size)
    train_dataset = ACDC(root=root, split="train", transform=train_transform)
    test_dataset = ACDC(root=root, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader




def get_ssl_acdc_loader(root=r'/home/ubuntu/data/ACDC', batch_size=8, unlabel_batch_size=24, train_crop_size=(224, 224), label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param unlabel_batch_size: 无标注数据的batch大小
    :param label_num: 有标签的数量
    :return:
    """
    train_transform=RandomGenerator(train_crop_size)
    train_dataset = ACDC(root=root, split="train", transform=train_transform)
    label_length = int(len(train_dataset) * label_num)
    train_label, train_unlabel = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[label_length, len(train_dataset) - label_length])

    test_dataset = ACDC(root=root, split="test")
    label_loader = DataLoader(train_label, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True )
    unlabel_loader = DataLoader(train_unlabel, batch_size=unlabel_batch_size, num_workers=4, shuffle=True,drop_last=True )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    return label_loader, unlabel_loader, test_loader

def show(im):
    im = im.numpy().squeeze()
    fig=plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()
    fig.savefig("result.png")


def show_label(mask, path="label.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).save(path)

# def get_ssl_acdc_loader(root=r'/home/ubuntu/data/ACDC', batch_size=8, unlabel_batch_size=24, train_crop_size=(224, 224), label_num=0.2,seed=1337):
#     """
#     :param root: 数据集路径
#     :param batch_size: 有标注数据批次大小
#     :param unlabel_batch_size: 无标注数据的batch大小
#     :param label_num: 有标签的数量
#     :return:
#     """

#     def worker_init_fn(worker_id):
#         random.seed(seed + worker_id)
    
#     train_transform=RandomGenerator(train_crop_size)
#     train_dataset = ACDC(root=root, split="train", transform=train_transform)
#     total_slices=len(train_dataset)
#     labeled_slice=patients_to_slices("ACDC",label_num)
#     labeled_idxs = list(range(0, labeled_slice))
#     unlabeled_idxs = list(range(labeled_slice, total_slices))

#     batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size+unlabel_batch_size,unlabel_batch_size)
#     train_loader=DataLoader(train_dataset,batch_sampler=batch_sampler,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)

#     test_dataset = ACDC(root=root, split="test")
#     test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
#     return train_loader, test_loader



if __name__ == '__main__':

    train_dataloader, test_dataloader = get_acdc_loader()
    # print(len(train_dataloader))
    # print(len(test_dataloader))
    # print(len(test_dataloader.dataset))
    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(train_dataloader.dataset.label_to_img(label))
        break

    for sample in test_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        break
