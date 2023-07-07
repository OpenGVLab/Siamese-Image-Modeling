# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MSN (https://github.com/facebookresearch/msn)
# Copyright (c) Facebook, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import argparse
import logging
import pprint

import numpy as np
import torch
import torchvision.transforms as transforms
import cyanure as cyan


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lambd', type=float,
    default=0.00025,
    help='regularization')
parser.add_argument(
    '--penalty', type=str,
    help='regularization for logistic classifier',
    default='l2',
    choices=[
        'l2',
        'elastic-net'
    ])
parser.add_argument(
    '--mask', type=float,
    default=0.0,
    help='regularization')
parser.add_argument(
    '--preload', action='store_true',
    help='whether to preload embs if possible')
parser.add_argument(
    '--fname', type=str,
    help='model architecture')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture')
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--image-folder', type=str,
    default='imagenet_full_size/061417/',
    help='image directory inside root_path')
parser.add_argument(
    '--subset-path', type=str,
    default=None,
    help='name of dataset to evaluate on')
parser.add_argument('--local_rank', default=-1, type=int)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    blocks,
    lambd,
    mask_frac,
    preload,
    pretrained,
    fname,
    subset_path,
    root_path,
    image_folder,
    penalty='l2',
    model_name=None,
    normalize=True,
    device_str='cuda:0',
    args=None
):
    init_distributed_mode(args)
    # torch.cuda.set_device(args.rank)
    # device = torch.device('cuda')
    # device = torch.device(device_str)
    # if 'cuda' in device_str:
    #     torch.cuda.set_device(device)

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None else 'imagenet_subses1-100percent'
    train_embs_path = f'train-features-{subset_tag}-{fname}'
    test_embs_path = f'val-features-{fname}'
    logger.info(train_embs_path)
    logger.info(test_embs_path)

    # pretrained = os.path.join(pretrained, fname)

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = subset_path if training else None
        data_loader, _ = init_data(
            transform=transform,
            batch_size=64,
            num_workers=0,
            world_size=args.world_size,
            rank=args.rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=False,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        # device=device,
        pretrained=pretrained,
        model_name=model_name)
    encoder.eval()

    # -- If train embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            # device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')
    # # -- Normalize embeddings
    cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)

    # import pdb; pdb.set_trace()

    # -- Fit Logistic Regression Classifier
    classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty=penalty, fit_intercept=False)
    lambd /= len(embs)
    classifier.fit(
        embs.numpy(),
        labs.numpy(),
        it0=10,
        lambd=lambd,
        lambd2=lambd,
        nthreads=-1,
        tol=1e-3,
        solver='auto',
        seed=0,
        max_epochs=300)

    # -- Evaluate and log
    train_score = classifier.score(embs.numpy(), labs.numpy())
    # -- (save train score)
    logger.info(f'train score: {train_score}')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        data_loader = init_pipe(False)
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            # device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

    # -- Evaluate and log
    test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
    # -- (save test score)
    logger.info(f'test score: {test_score}\n\n')

    return test_score


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.rank % num_gpus)
        import subprocess
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        # specify master port
        if hasattr(args, 'port'):
            os.environ['MASTER_PORT'] = str(args.port)
        elif 'MASTER_PORT' in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # 29500 is torch.distributed default port
            os.environ['MASTER_PORT'] = '29502'
        # use MASTER_ADDR in the environment variable if it already exists
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.rank % num_gpus)
        os.environ['RANK'] = str(args.rank)
        # dist.init_process_group(backend='nccl')
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def init_data(
    transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):

    # dataset = ImageNet(
    #     root=root_path,
    #     image_folder=image_folder,
    #     transform=transform,
    #     train=training,
    #     copy_data=copy_data)
    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)
    import torchvision
    if training:
        dataset = torchvision.datasets.ImageFolder(os.path.join(root_path, 'train'), transform=transform)
        with open(subset_file) as subset_file:
            list_imgs = [li.split('\n')[0] for li in subset_file.readlines()]
        dataset.samples = [(
            os.path.join(os.path.join(root_path, 'train'), li.split('_')[0], li),
            dataset.class_to_idx[li.split('_')[0]]
        ) for li in list_imgs]
    else:
        dataset = torchvision.datasets.ImageFolder(os.path.join(root_path, 'val'), transform=transform)

    logger.info('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    logger.info('ImageNet unsupervised data loader created')

    return (data_loader, dist_sampler)


def make_embeddings(
    blocks,
    # device,
    mask_frac,
    data_loader,
    encoder,
    epochs=1
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.cuda()
            with torch.no_grad():
                z = encoder.forward_features(imgs)[:, 0].cpu()
            labels = labels.cpu()
            z_mem.append(z)
            l_mem.append(labels)
            if itr % 50 == 0:
                logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    z_mem = all_gather(z_mem)
    z_mem = torch.cat(z_mem, 0)
    l_mem = all_gather(l_mem)
    l_mem = torch.cat(l_mem, 0)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)

    return z_mem, l_mem


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    import pickle
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def load_pretrained(
    encoder,
    pretrained
):
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    try:
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {pretrained}')
    except Exception:
        pass
    del checkpoint
    return encoder


def init_model(
    # device,
    pretrained,
    model_name,
):
    # encoder = deit.__dict__[model_name]()
    # encoder.fc = None
    # encoder.to(device)
    # encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    import models_vit
    model = models_vit.__dict__[model_name](
        num_classes=1000,
        global_pool=True,
        init_values=None,
        drop_path_rate=0.0
    )

    checkpoint = torch.load(pretrained, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % pretrained)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    model.head = None

    model.cuda()

    return model


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    global args
    args = parser.parse_args()
    pp.pprint(args)
    main(
        blocks=1,
        lambd=args.lambd,
        penalty=args.penalty,
        mask_frac=args.mask,
        preload=args.preload,
        pretrained=args.pretrained,
        fname=args.fname,
        subset_path=args.subset_path,
        root_path=args.root_path,
        image_folder=args.image_folder,
        model_name=args.model_name,
        normalize=args.normalize,
        device_str=args.device,
        args=args
    )
