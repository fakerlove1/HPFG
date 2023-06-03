import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, mask):
        if random.random() > 0.5:
            image, mask = random_rot_flip(image, mask)
        elif random.random() > 0.5:
            image, mask = random_rotate(image, mask)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        mask = zoom(
            mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.uint8))
        sample = {'image': image, 'mask': mask}
        return sample


class Synapse(Dataset):

    PALETTE = np.array([[0, 0, 0], [0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
                        [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
                        [192, 128, 128]])

    def __init__(self, root, split, transform=None):
        super(Synapse, self).__init__()
        self.split = split
        self.root = root
        self.transform = transform
        self.sample_list = []
        self.load_annotations()  # 加载文件路径
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        filepath = self.sample_list[idx]
        if self.split == "train":
            data = np.load(filepath)
            image, mask = data['image'], data['label']
        else:
            data = h5py.File(filepath)
            image, mask = data['image'][:], data['label'][:]

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
        return image, mask

    def load_annotations(self):
        if self.split == "train":
            with open(self.root + "/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [os.path.join(self.root, "train_npz", line+".npz") for line in self.sample_list]
        else:
            with open(self.root + "/test_vol.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [os.path.join(self.root, "test_vol_h5", line+".npy.h5") for line in self.sample_list]

        self.sample_list = np.array(self.sample_list)

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


def get_synapse_loader(root=r'/home/ubuntu/data/Synapse', batch_size=8, train_crop_size=(224, 224)):
    # train_transform = A.Compose([
    #     A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1],scale=(0.5, 2.0)),
    #     A.ShiftScaleRotate(p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.8),
    #     A.ColorJitter(0.4,0.4,0.4,p=0.5),
    #     ToTensorV2()
    # ])
    train_transform = RandomGenerator(train_crop_size)
    train_dataset = Synapse(root=root, split="train", transform=train_transform)
    test_dataset = Synapse(root=root, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


def get_ssl_synapse_loader(root=r'/home/ubuntu/data/Synapse', batch_size=8, unlabel_batch_size=24, train_crop_size=(224, 224), label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param unlabel_batch_size: 无标注数据的batch大小
    :param label_num: 有标签的数量
    :return:
    """
    train_transform = RandomGenerator(train_crop_size)
    train_dataset = Synapse(root=root, split="train", transform=train_transform)
    label_length = int(len(train_dataset) * label_num)
    train_label, train_unlabel = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[label_length, len(train_dataset) - label_length])

    test_dataset = Synapse(root=root, split="test")
    label_loader = DataLoader(train_label, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    unlabel_loader = DataLoader(train_unlabel, batch_size=unlabel_batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    return label_loader, unlabel_loader, test_loader


def show(im):
    im = im.numpy().squeeze()
    fig = plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()
    fig.savefig("result.png")


def show_label(mask, path="label.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).convert("RGB").save(path)


if __name__ == '__main__':

    train_dataloader, test_dataloader = get_synapse_loader()
    # for image, label in train_dataloader:
    #     print(image.shape)
    #     print(label.shape)
    #     print(np.unique(label.numpy()))
    #     show(image[0])
    #     show_label(train_dataloader.dataset.label_to_img(label[0]))
    #     break
    total=0
    for sample in test_dataloader:
        image, label = sample
        total+=image.shape[1]
        print(image.shape)
        # print(label.shape)
        # print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        # break
    # print(total)
