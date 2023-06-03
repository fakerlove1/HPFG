# 🦕HPFG

> Official PyTorch implementation of "HPFG: Semi-Supervised Medical Image Segmentation Framework based on Hybrid Pseudo-Labeling and Feature-Guided"



## 🏷️Support

Currently, we have implemented 7 popular semi supervised medical image segmentation algorithms.

| Date    | Name         | Title                                                        | Reference |
| ------- | ------------ | ------------------------------------------------------------ | --------- |
| 2017-03 | Mean-Teacher | [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780) | NeurlPS   |
| 2019-07 | UAMT         | [Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation (arxiv.org)](https://arxiv.org/abs/1907.07034) | MICCAI    |
| 2021-06 | CPS          | [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226) | CVPR      |
| 2021-12 | CTCT         | [Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer](https://arxiv.org/abs/2112.0489) | MIDL      |
| 2022-02 | ICT-MegSeg   | [AN EMBARRASSINGLY SIMPLE CONSISTENCY REGULARIZATION METHOD FOR SEMI-SUPERVISED MEDICAL IMAGE SEGMENTATION](https://arxiv.org/abs/2202.00677) | ISBI      |
| 2022-03 | SSNet        | [Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/2203.01324v3) | MICCAI    |
| 2022-08 | S4CVNet      | [When CNN Meet with ViT: Towards Semi-Supervised Learning for Multi-Class Medical Image Semantic Segmentation](https://arxiv.org/abs/2208.06449) | CVPR      |



## 🛠️ Install 

If it is a pip environment, run the following command

~~~bash
pip install -r requirements.txt
~~~

If it is a Conda environment, run the following command

~~~bash
conda env create -f requirements.yml
~~~



## ✨ DataSets

| DataSets | Downloadlink                                         |
| -------- | ---------------------------------------------------- |
| ACDC     | https://www.kaggle.com/datasets/jokerak/acdch5       |
| LIDC     | https://www.kaggle.com/datasets/jokerak/lidcidri     |
| ISIC     | https://www.kaggle.com/datasets/jokerak/isic2018-224 |



## ⭐Train

`step 1`: Download the code and prepare the running environment

1. Clone this repo to your machine.
2. Make sure Anaconda or Miniconda is installed.
3. Run `pip install -r requirement.txt` for environment initialization.



`step 2`: Download Datasets



`step 3`: It is convenient to perform experiment with HPFG. For example, if you want to run Mean-Teacher algorithm:

1. Modify the config file in `config/mean_teacher_unet_30k_224x224_ACDC.yaml` as you need

   ~~~yaml
   # Dataset Configuration
   datasets: "acdc" # Dataset name
   num_classes: 4 # Number of categories
   data_path: "/home/ubuntu/data/ACDC" # Dataset placement location
   save_path: "checkpoint/2023-02-26-mean_teacher-ACDC" # Code Save Location
   name: "mean_teacher-ACDC"
   ckpt: None # Pre-training weight position
   cuda: True # Whether to use GPU
   ~~~

2. Run `python 2017_03_NIPS_Mean-Teacher_ACDC.py`



If you want to run the paper project, please run the following code directly

~~~python
python main.py
~~~



## ♥️ Acknowledgement

Our model is related to [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for their great work!