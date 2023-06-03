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
import os


class Building(Dataset):

    PALETTE = np.array([
        [0, 0, 0],
        [255, 255, 255],
    ])

    def __init__(self, root=r"E:\note\ssl\data\ACDC", split="train", transform=None, index=None):

        super(Building, self).__init__()
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

        if self.split == "train" or self.split == "val":
            image = np.array(Image.open(self.img_dir[idx]).convert("RGB"), dtype=np.float32)
            mask = np.array(Image.open(self.ann_dir[idx]).convert("L"), dtype=np.uint8)
            image = image.astype('float32') / 255
            mask[mask == 255] = 1

            if self.transform is not None:
                result = self.transform(image=image, mask=mask)
                image = result["image"]
                mask = result["mask"]

            return image, mask
        else:
            image = np.array(Image.open(self.img_dir[idx]).convert("RGB"), dtype=np.float32)
            image = image.astype('float32') / 255
            return self.transform(image=image)["image"]

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
            img = make_grid(tensor=img, nrow=2, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

    def load_annotations(self):
        if self.split == "train":
            with open(self.root + "/train.txt", "r") as f:
                self.sample_list = f.readlines()
            self.root=os.path.join(self.root,"train")
        elif self.split == "val":
            with open(self.root + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.root=os.path.join(self.root,"train")
        else:
            with open(self.root + "/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.root=os.path.join(self.root,"test")

        self.sample_list = [item.replace("\n", "") for item in self.sample_list]
    

        self.img_dir = [os.path.join(self.root, "image", item) for item in self.sample_list]
        self.ann_dir = [os.path.join(self.root, "mask", "{}.png".format(item.split(".")[0])) for item in self.sample_list]
        self.img_dir = np.array(self.img_dir)
        self.ann_dir = np.array(self.ann_dir)


def get_building_loader(root=r'/home/ubuntu/data/Ali_building_2class', 
                        batch_size=8, 
                        train_crop_size=(512, 512), 
                        ):
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
        A.ColorJitter(0.4, 0.4, 0.4, p=0.5),
        # A.Normalize(),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        # A.Resize(height=train_crop_size[0], width=train_crop_size[1]),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    train_dataset = Building(root=root, split="train", transform=train_transform)
    val_dataset = Building(root=root, split="val", transform=test_transform)
    test_dataset = Building(root=root, split="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, val_loader, test_loader


def show(im):
    im = im.permute(1, 2, 0).numpy()
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

    train_loader, val_loader, test_loader = get_building_loader()
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(test_loader.dataset.label_to_img(label))
        break

    for sample in val_loader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        break
