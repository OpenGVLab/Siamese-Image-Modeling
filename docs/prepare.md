# Preparation

* The only dataset required in this repo is ImageNet, which is enough for pretraining, finetuning, linear evaluation and few-shot evaluation. If you want to evaluate on COCO, LVIS, ADE20k and robustness datasets, please follow the corresponding repos to prepare the data.

## Installation

* Python >=3.7
* We recommend to use Pytorch1.11 for a faster training speed.
* timm == 0.6.12

To run few-shot evaluation, [cyanure](https://github.com/inria-thoth/cyanure) package is further required. You can install it with
```
  pip install cyanure-openblas
  # or pip install cyanure-mkl
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
  /path/to/imagenet/
      ├── train/
      │   ├── class1/
      │   │   ├── img1.JPEG
      |   │   ├── img2.JPEG
      |   │   ├── img3.JPEG
      |   │   └── ...
      │   ├── class2/
      |   │   └── ...   
      │   ├── class3/
      |   │   └── ...
      |   └── ...
      └─── val
      │   ├── class1/
      │   │   ├── img4.JPEG
      |   │   ├── img5.JPEG
      |   │   ├── img6.JPEG
      |   │   └── ...
      │   ├── class2/
      |   │   └── ...   
      │   ├── class3/
      |   │   └── ...
```

Note that raw val images are not put into class folders, use [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) to get correct layout.
