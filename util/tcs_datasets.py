# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import io
from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa
import numpy as np
from io import BytesIO
import tqdm
from tqdm import trange
try:
    from petrel_client.client import Client
except ImportError as E:
    "petrel_client.client cannot be imported"
    pass


def tcs_pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


class TCSLoader(object):

    def __init__(self, conf_path):
        self.client = Client(conf_path)

    def __call__(self, fn):
        try:
            img_value_str = self.client.get(fn)
            img = tcs_pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img


def _get_images(annotations):
    images = []
    classes = []
    for line in annotations:
        if isinstance(line, bytes):
            line = line.decode()
        image_name, cls = line.strip('\n').split()
        images.append(image_name)
        classes.append(cls)
    return images, classes


class ImageNetTCSDatasetQK(Dataset):
    def __init__(self, image_set, data_path, transform=None, use_tcs=False,
                 tcs_conf_path='/mnt/lustre/share_data/taochenxin/tcs/petreloss.conf',
                 test_mode=False, 
                 on_memory=False, local_rank=None, local_size=None,
                 **kwargs):
        ann_file = os.path.join(data_path, f'meta/{image_set}.txt')
        data_path = os.path.join(data_path, image_set)
        self.image_set = image_set
        self.transform = transform
        self.data_path = data_path
        self.test_mode = test_mode
        if use_tcs:
            self.tcs_loader = TCSLoader(tcs_conf_path)
        self.use_tcs = use_tcs
        self.images, self.classes, self.class_to_idx = self._load_database(ann_file)
        self.on_memory = on_memory
        if on_memory:
            if local_rank is None:
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_size is None:
                local_size = int(os.environ.get('LOCAL_SIZE', 1))
            self.local_rank = local_rank
            self.local_size = local_size
            self.holder = {}
            self.load_onto_memory()

    def load_onto_memory(self):
        print("Loading images onto memory...")
        for index in trange(len(self.images)):
            if index % self.local_size != self.local_rank:
                continue
            path = self.images[index].as_py()
            full_path = os.path.join(self.data_path, path)
            if self.use_tcs:
                sample = self.tcs_loader.client.get(full_path)
            else:
                with open(full_path, 'rb') as f:
                    sample = f.read()
            self.holder[path] = sample
        # print('Loading: path {}, full_path {}, data length {}'.format(path, full_path, 
        #                                                               len(self.tcs_loader.client.get(full_path))))
        print("Loading complete!")

    def _load_database(self, annotation_file):
        if not self.use_tcs:
            annotation_file = os.path.abspath(annotation_file)
        print(f'loading annotations from {annotation_file} ...')
        if self.use_tcs:
            with BytesIO(self.tcs_loader.client.get(annotation_file)) as annotations:
                images, classes = _get_images(annotations)
        else:
            with open(annotation_file, 'rt') as annotations:
                images, classes = _get_images(annotations)

        # convert possible classes to indices
        class_names = sorted(set(classes))
        # class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        class_to_idx = {class_name: int(class_name) for class_name in class_names}
        return pa.array(images), pa.array([class_to_idx[class_name] for class_name in classes]), class_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index].as_py()
        target = self.classes[index].as_py()
        sample = self._load_image(path)
        if self.transform is not None:
            sample_q = self.transform(sample)
            sample_k = self.transform(sample)
            return sample_q, sample_k
        else:
            return sample, sample

    def _load_image(self, path):
        full_path = os.path.join(self.data_path, path)
        if self.on_memory:
            try:
                return Image.open(BytesIO(self.holder[path])).convert('RGB')
            except:
                print('error acquiring data from {}'.format(path))
                return self.tcs_loader(full_path).convert('RGB')
        elif self.use_tcs:
            return self.tcs_loader(full_path).convert('RGB')
        else:
            with open(full_path, 'rb') as f:
                return Image.open(f).convert('RGB')


class ImagenetTCSDataset(ImageNetTCSDatasetQK):
    def __init__(self, image_set, data_path, transform=None, use_tcs=False, 
                 tcs_conf_path='/mnt/lustre/share_data/taochenxin/tcs/petreloss.conf', 
                 test_mode=False, on_memory=False, local_rank=None, local_size=None, 
                 with_blockwise_mask=False, ### !!! set to True, enable blockwise masking
                 blockwise_num_masking_patches=75, ### !!! 75 / 196 = 0.38 -> Modify this to increase mask ratio
                 input_size=224, patch_size=16, # no need to change now
                 max_mask_patches_per_block=None, # BEiT default setting, no need to change
                 min_mask_patches_per_block=16, # BEiT default setting, no need to change
                 fixed_num_masking_patches=True, ### set to true, fixed number of masking patch to blockwise_num_masking_patches for sim training 
                 **kwargs):
        super().__init__(image_set, data_path, transform=transform, use_tcs=use_tcs, 
                         tcs_conf_path=tcs_conf_path, test_mode=test_mode, on_memory=on_memory, 
                         local_rank=local_rank, local_size=local_size, **kwargs)
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
        path = self.images[index].as_py()
        target = self.classes[index].as_py()
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.with_blockwise_mask:
            return sample, target, self.masked_position_generator()
        return sample, target


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = ImagenetTCSDataset(
        'val',
        's3://imagenet', 
        tcs_conf_path='./petreloss.conf',
        transform=transform,
        with_blockwise_mask=True,
        blockwise_num_masking_patches=75)
    for i, (sample, target, mask) in enumerate(dataset):
        if i < 10:
            print(mask.sum())
            print(mask)
        else:
            break
