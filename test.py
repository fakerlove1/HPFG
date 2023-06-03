import numpy as np
import pandas as pd
import cv2
import os
# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


import torch
from tqdm import tqdm
from model import build_model
from datasets import build_loader
from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss, Dense_Loss, BoxMaskGenerator,SegMetrics
path = r"config/ccnet_unet_80k_100%_512x512_Building.yaml"
root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
args = loadyaml(os.path.join(root, path))  # 加载yaml
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
# args.save_path = os.path.join(root, args.save_path)
# mk_path(args.save_path)  # 创建文件保存位置
# # 创建 tensorboardX日志保存位置
# mk_path(os.path.join(args.save_path, "tensorboardX"))
# mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
# args.model_save_path = os.path.join(args.save_path, "model", "model.pth")

# train_loader, val_loader,test_loader = build_loader(args)  # 构建数据集
model = build_model(args=args).to(device=args.device)  # 创建模型




checkpoint=torch.load(r"/home/ubuntu/code/pytorch_code/code/2-semi-medseg/checkpoint/2023-03-16_pretrain_unet_plus_80k_512x512_ISIC/model/ema_model_model.pth")
model.load_state_dict(checkpoint["model"],strict=True)
test_mask = pd.read_csv('/home/ubuntu/code/pytorch_code/code/2-semi-medseg/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])

test_mask['name'] = test_mask['name'].apply(lambda x: '/home/ubuntu/data/Ali_building_2class/test/image/' + x)

from torchvision import transforms as T

trfm = T.Compose([
    # T.ToPILImage(),
    # T.Resize(512),
    T.ToTensor(),
#     T.Normalize([0.625, 0.448, 0.688],
#                 [0.131, 0.177, 0.101]),
])

print(test_mask)
from PIL import Image
model.eval()
subm = []
for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
    image = Image.open(name).convert("RGB")
    image = trfm(image)
    with torch.no_grad():
        image = image.to(args.device).unsqueeze(0)
        print(image.shape)
        out = model.val(image)
        label_pred = out.detach().max(dim=1)[1].data.cpu().numpy()

        subm.append([name.split('/')[-1], rle_encode(label_pred)])

        
subm = pd.DataFrame(subm)
subm.to_csv('/home/ubuntu/code/pytorch_code/code/2-semi-medseg/tmp.csv', index=None, header=None, sep='\t') 

        

