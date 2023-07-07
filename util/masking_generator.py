# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None, fixed_num_masking_patches=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.fixed_num_masking_patches = fixed_num_masking_patches

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        
        if self.fixed_num_masking_patches and (mask_count < self.num_masking_patches):
            non_masked_inds_i, non_masked_inds_j  = (mask == 0).nonzero()
            shuffle_inds = list(range(non_masked_inds_i.shape[0]))
            random.shuffle(shuffle_inds)
            num_to_mask = self.num_masking_patches - mask_count
            to_mask_inds_i = non_masked_inds_i[shuffle_inds[:num_to_mask]]
            to_mask_inds_j = non_masked_inds_j[shuffle_inds[:num_to_mask]]
            mask[to_mask_inds_i, to_mask_inds_j] = 1
            mask_count += num_to_mask
        
        return mask


if __name__ == '__main__':
    blockwise_num_masking_patches=75 ### TODO: 75 / 196 = 0.38 -> Modify this to increase mask ratio
    input_size=224
    patch_size=16 # BEiT default setting, no need to change
    max_mask_patches_per_block=None # BEiT default setting, no need to change
    min_mask_patches_per_block=16 # BEiT default setting, no need to change
    fixed_num_masking_patches=True ### TODO: fixed number of masking patch to blockwise_num_masking_patches for sim training
    window_size = input_size // patch_size
    masked_position_generator = MaskingGenerator(
        (window_size, window_size),
        num_masking_patches=blockwise_num_masking_patches,
        max_num_patches=max_mask_patches_per_block,
        min_num_patches=min_mask_patches_per_block,
        fixed_num_masking_patches=fixed_num_masking_patches
    )
    mask_num = []
    for _ in range(10000):
        mask = masked_position_generator()
        if _ < 10:
            print(mask)
        mask_num.append(mask.sum())
    print(f"Max Patches: {max(mask_num)} Min Patches: {min(mask_num)} Mean Patches: {sum(mask_num) / len(mask_num)}")
    print(f"Max Ratio: {max(mask_num)/196.0} Min Ratio: {min(mask_num)/196.0} Mean Ratio: {sum(mask_num) / len(mask_num) / 196.0}")
