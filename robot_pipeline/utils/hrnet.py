# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import sys
from os import path

print(path.join(path.dirname(path.dirname(path.abspath(__file__))),
                '/home/fanfan/Documents/deep-high-resolution-net.pytorch/lib/'))
sys.path.append(path.join(path.dirname(path.dirname(path.abspath(__file__))),
                          '/home/fanfan/Documents/deep-high-resolution-net.pytorch/lib/'))
sys.path.append('/home/fanfan/Documents/deep-high-resolution-net.pytorch/tools')

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import cv2
import dataset
import models
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        type=str,
                        default='/home/fanfan/Documents/deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
                + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
                shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1

    return center, scale


def hrnet_load(self):
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    self.model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            '/home/fanfan/Documents/deep-high-resolution-net.pytorch/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        self.model.load_state_dict(torch.load(model_state_file))
    self.model = torch.nn.DataParallel(self.model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()


def hrnet_predict(self):
    data_numpy = self.rotate_cv_image

    # object detection box
    box = [0, 0, 720, 1280]
    # box = [self.pt1[0] - 10, self.pt1[1], 490 - self.pt1[0], 640 - self.pt1[1]]
    c, s = _box2cs(box, data_numpy.shape[1], data_numpy.shape[0])
    r = 0

    trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input = transform(input).unsqueeze(0)

    # switch to evaluate mode
    self.model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = self.model(input)
        # compute coordinate
        self.preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

        # # plot
        # image = data_numpy.copy()
        # for mat in self.preds[0]:
        #     x, y = int(mat[0]), int(mat[1])
        #     cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
        #
        # cv2.line(image, (self.preds[0][10][0], self.preds[0][10][1]), (self.preds[0][8][0], self.preds[0][8][1]),
        #          (255, 255, 0), 5)
        # cv2.line(image, (self.preds[0][6][0], self.preds[0][6][1]), (self.preds[0][8][0], self.preds[0][8][1]),
        #          (255, 0, 0), 5)
        # cv2.line(image, (self.preds[0][7][0], self.preds[0][7][1]), (self.preds[0][5][0], self.preds[0][5][1]),
        #          (255, 0, 0), 5)
        # cv2.line(image, (self.preds[0][7][0], self.preds[0][7][1]), (self.preds[0][9][0], self.preds[0][9][1]),
        #          (255, 0, 0), 5)
        # cv2.imshow('res', image)
        # cv2.waitKey(1)
