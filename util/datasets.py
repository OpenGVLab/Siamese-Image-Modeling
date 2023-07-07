# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImagenetWithMask(datasets.ImageFolder):
    def __init__(self, root,
                transform = None,
                with_blockwise_mask=False, ### !!! set to True, enable blockwise masking
                 blockwise_num_masking_patches=75, ### !!! 75 / 196 = 0.38 -> Modify this to increase mask ratio
                 input_size=224, patch_size=16, # no need to change now
                 max_mask_patches_per_block=None, # BEiT default setting, no need to change
                 min_mask_patches_per_block=16, # BEiT default setting, no need to change
                 fixed_num_masking_patches=True, ### set to true, fixed number of masking patch to blockwise_num_masking_patches for sim training 
                 ):
        super().__init__(root, transform)
        self.with_blockwise_mask = with_blockwise_mask
        if with_blockwise_mask:
            from .masking_generator import MaskingGenerator
            window_size = input_size // patch_size
            self.masked_position_generator = MaskingGenerator(
                (window_size, window_size), 
                num_masking_patches=blockwise_num_masking_patches,
                max_num_patches=max_mask_patches_per_block,
                min_num_patches=min_mask_patches_per_block,
                fixed_num_masking_patches=fixed_num_masking_patches
            )
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if self.with_blockwise_mask:
            return sample, target, self.masked_position_generator()
        return sample, target
