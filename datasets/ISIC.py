import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from scipy import ndimage
from skimage import io
from albumentations.augmentations import transforms
from scipy.ndimage import zoom
import random


class ISIC(Dataset):

    PALETTE = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ])

    def __init__(self, root=r"E:\note\ssl\data\ACDC", split="train", transform=None, index=None):

        super(ISIC, self).__init__()
        self.split = split
        self.root = root
        self.transform = transform
        self.img_dir = []
        self.ann_dir = []
        self.load_annotations()  # 加载文件路径
        print("total {} samples".format(len(self.img_dir)))

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.img_dir[idx]).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(self.ann_dir[idx]).convert("L"), dtype=np.uint8)
        image = image.astype('float32') / 255
        mask[mask > 0] = 1

        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        return image, mask

    def label_to_img(self, label):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=8, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

    def load_annotations(self):
        if self.split == "train":
            with open(self.root + "/train.txt", "r") as f:
                self.sample_list = f.readlines()
        else:
            with open(self.root + "/test.txt", "r") as f:
                self.sample_list = f.readlines()

        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        self.img_dir = [self.root + "/image/{}.jpg".format(item) for item in self.sample_list]
        self.ann_dir = [self.root + "/gt/{}_segmentation.png".format(item) for item in self.sample_list]

        self.img_dir = np.array(self.img_dir)
        self.ann_dir = np.array(self.ann_dir)


def get_isic_loader(root=r'/home/ubuntu/data/ISIC2018_224', batch_size=2, train_crop_size=(224, 224)):
    """
    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1], scale=(0.75, 1.5)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.6),
        A.ColorJitter(0.4, 0.4, 0.4, p=0.5),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(train_crop_size[0], train_crop_size[1]),
        ToTensorV2()
    ])

    train_dataset = ISIC(root=root, split="train", transform=train_transform)
    test_dataset = ISIC(root=root, split="test", transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


def get_ssl_isic_loader(root=r'/home/ubuntu/data/ISIC2018_224',
                        batch_size=8,
                        unlabel_batch_size=24,
                        train_crop_size=(224, 224),
                        label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param unlabel_batch_size: 无标注数据的batch大小
    :param label_num: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1], scale=(0.5, 2.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.6),
        A.RandomBrightnessContrast(p=0.2),
        # A.ColorJitter(0.4, 0.4, 0.4, p=0.5),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(height=train_crop_size[0], width=train_crop_size[1]),
        ToTensorV2()
    ])

    train_dataset = ISIC(root=root, split="train", transform=train_transform)
    label_length = int(len(train_dataset) * label_num)
    train_label, train_unlabel = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[label_length, len(train_dataset) - label_length])

    test_dataset = ISIC(root=root, split="test", transform=test_transform)
    label_loader = DataLoader(train_label, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    unlabel_loader = DataLoader(train_unlabel, batch_size=unlabel_batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return label_loader, unlabel_loader, test_loader


def show(im):
    im = im.permute(1, 2, 0).numpy()
    # image=Image.fromarray(im).convert('RGB')
    # image.save("result.jpg")

    fig = plt.figure()
    plt.imshow(im)
    plt.show()
    fig.savefig("result.jpg")


def show_label(mask, path="label.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).save(path)


if __name__ == '__main__':

    train_dataloader, test_dataloader = get_isic_loader()
    print(len(train_dataloader.dataset))
    print(len(test_dataloader.dataset))
    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(train_dataloader.dataset.label_to_img(label))
        break

    for sample in test_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        break
