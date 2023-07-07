# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
assert timm.__version__ == "0.6.12"  # version check
from timm.optim.optim_factory import param_groups_weight_decay
from timm.optim import create_optimizer

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.augmentation import RandomResizedCrop, GaussianBlur, SingleRandomResizedCrop, RandomHorizontalFlip, Solarize
from util.datasets import ImagenetWithMask
import models_sim
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--use_abs_pos_emb', default=True, action='store_true')
    parser.add_argument('--disable_abs_pos_emb', dest='use_abs_pos_emb', action='store_false')
    parser.add_argument('--use_shared_rel_pos_bias', default=False, action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # SiameseIM parameters
    # data
    parser.add_argument('--crop_min', default=0.2, type=float)
    parser.add_argument('--use_tcs_dataset', default=False, action='store_true')

    # model
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--init_values', default=None, type=float)
    parser.add_argument('--projector_depth', default=2, type=int)
    parser.add_argument('--predictor_depth', default=4, type=int)
    parser.add_argument('--use_proj_ln', default=False, action='store_true')
    parser.add_argument('--use_pred_ln', default=False, action='store_true')
    parser.add_argument('--train_patch_embed', default=False, action='store_true')
    parser.add_argument('--online_ln', default=False, action='store_true', help='also use frozen LN in online branch')
    
    parser.add_argument('--loss_type', default='mae')
    parser.add_argument('--neg_weight', default=0.02, type=float)

    parser.add_argument('--with_blockwise_mask', default=False, action='store_true')
    parser.add_argument('--blockwise_num_masking_patches', default=75, type=int)

    # hyper-parameter
    parser.add_argument('--mm', default=0.996, type=float)
    parser.add_argument('--mmschedule', default='const')
    parser.add_argument('--lambda_F', default=50, type=float) # may no need
    parser.add_argument('--T', default=0.2, type=float)       # check
    parser.add_argument('--clip_grad', default=None, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)

    # misc
    parser.add_argument('--auto_resume', default=True)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--save_latest_freq', default=1, type=int)
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--amp_growth_interval', default=2000, type=int)

    return parser


class DataAugmentationForSIM(object):
    def __init__(self, args):
        self.args = args

        self.random_resized_crop = SingleRandomResizedCrop(args.input_size, scale=(args.crop_min, 1.0), interpolation=3)
        self.random_flip = RandomHorizontalFlip()

        self.color_transform1 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        ])

        self.color_transform2 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
        ])

        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        spatial_image1, flip1 = self.random_flip(image)
        spatial_image2, flip2 = self.random_flip(image)
        spatial_image1, i1, j1, h1, w1, W = self.random_resized_crop(spatial_image1)
        spatial_image2, i2, j2, h2, w2, W = self.random_resized_crop(spatial_image2)
        color_image1 = self.color_transform1(spatial_image1)
        color_image2 = self.color_transform2(spatial_image2)

        relative_flip = (flip1 and not flip2) or (flip2 and not flip1)
        return self.format_transform(color_image1), self.format_transform(color_image2), \
                (i2-i1)/h1, (j2-j1)/w1, h2/h1, w2/w1, relative_flip, (W-j1-j2)/w1

    def __repr__(self):
        repr = "(DataAugmentation,\n"
        repr += "  transform = %s,\n" % str(self.random_resized_crop) + str(self.random_flip) + str(self.color_transform1) + str(self.format_transform)
        repr += ")"
        return repr


def main(args):
    misc.init_distributed_mode(args) # need change to torch.engine

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    # disable tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # build augmentation and dataset
    if args.loss_type in ['sim']:
        transform_train = DataAugmentationForSIM(args)
    else:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not args.use_tcs_dataset:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_train = ImagenetWithMask(os.path.join(args.data_path, 'train'),
                                         transform=transform_train,
                                         with_blockwise_mask=args.with_blockwise_mask,
                                        blockwise_num_masking_patches=args.blockwise_num_masking_patches)
    else: # for internal use only
        from util.tcs_datasets import ImagenetTCSDataset
        dataset_train = ImagenetTCSDataset('train',
                                        's3://imagenet',
                                        use_tcs=True,
                                        transform=transform_train,
                                        with_blockwise_mask=args.with_blockwise_mask,
                                        blockwise_num_masking_patches=args.blockwise_num_masking_patches,
                                        local_rank=int(os.environ['LOCAL_RANK']),
                                        local_size=int(os.environ['LOCAL_SIZE']),
                                        tcs_conf_path='./petreloss.conf')
    print(dataset_train)

    # build dataloader
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    
    # build model
    model = models_sim.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, args=args)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    print("Model = %s" % str(model_without_ddp))
    
    # build optimizer
    # following timm: set wd as 0 for bias and norm layers
    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, args.beta2))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=(not args.fp32), growth_interval=args.amp_growth_interval)

    misc.auto_load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                         loss_scaler=loss_scaler)

    # start training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        dist.barrier()

        # save ckpt
        if args.output_dir and ((epoch+1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if (epoch+1) % args.save_latest_freq == 0:
            misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, latest=True)

        # log information
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if misc.is_main_process():
            epoch_total_time = time.time() - epoch_start_time
            now = datetime.datetime.today()
            eta = now + datetime.timedelta(seconds=(args.epochs-epoch-1)*int(epoch_total_time))
            next_50_ep = ((epoch + 1) // 50 + 1) * 50
            eta_to_next_50 =now + datetime.timedelta(seconds=(next_50_ep - epoch - 1) * int(epoch_total_time))
            print(f"ETA to {args.epochs:4d}ep:\t{str(eta)}")
            print(f"ETA to {next_50_ep:4d}ep:\t{str(eta_to_next_50)}")
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
