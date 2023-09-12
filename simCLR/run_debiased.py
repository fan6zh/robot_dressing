# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

import argparse
import torch
import sys
import time
import os
import torch.backends.cudnn as cudnn
import torchvision
import torchvideo
from torchvision import models
from data import datasets, transforms
# from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr_debiased import SimCLR

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mvit',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--tau_plus', default=0.2, type=float, help='Positive class priorx')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True

    # dataset = ContrastiveLearningDataset(args.data)
    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    ############################################################################################################ train
    # dataset = datasets.VideoDataset(
    #     "./data/flag_example_video_file.csv",
    #     transform=torchvision.transforms.Compose([
    #         # transforms.VideoFilePathToTensor(max_len=40, fps=2, padding_mode='last'),
    #         transforms.VideoFolderPathToTensor(max_len=30, padding_mode='last'),
    #         transforms.VideoResize([56, 56], torchvision.transforms.InterpolationMode.BILINEAR),
    #         transforms.VideoGrayscale(num_output_channels=3),
    #     ])
    # )
    #
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.workers, pin_memory=True, drop_last=True)

    ############################################################################################################ train
    dataset = datasets.VideoDataset(
        "./data/flag_example_video_file.csv",
        transform=torchvision.transforms.Compose([
            # transforms.VideoFilePathToTensor(max_len=40, fps=2, padding_mode='last'),
            transforms.VideoFolderPathToTensor(max_len=30, padding_mode='last'),
            transforms.VideoResize([56, 56], torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.VideoGrayscale(num_output_channels=3),
        ])
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = train_loader

    ############################################################################################################
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        # simclr.train(train_loader)
        simclr.test(test_loader)


if __name__ == "__main__":
    main()
