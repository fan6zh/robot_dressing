# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on deep-high-resolution-net.pytorch.
# (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
from os import path

sys.path.append('/home/fanfan/Documents/DEKR-main/lib/')
sys.path.append('/home/fanfan/Documents/DEKR-main/tools')

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate
from scipy import ndimage

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str,
                        default='/home/fanfan/Documents/DEKR-main/experiments/coco/inference_demo_coco.yaml')
    parser.add_argument('--outputDir', type=str, default='/home/fanfan/Documents/DEKR-main/out/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0.01)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def dekr_load(self):
    # transformation
    self.pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    self.args = parse_args()
    update_config(cfg, self.args)

    self.pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    self.pose_model.load_state_dict(torch.load(
        '/home/fanfan/Documents/DEKR-main/model/pose_coco/pose_dekr_hrnetw32_coco.pth'),
        strict=False)

    self.pose_model.to(CTX)
    self.pose_model.eval()

def dekr_predict(self):
    image_rgb = self.rotate_cv_image
    image_pose = image_rgb.copy()

    self.preds = get_pose_estimation_prediction(
        cfg, self.pose_model, image_pose, self.args.visthre, transforms=self.pose_transform)

    new_csv_row = []
    for coords in self.preds:
        # Draw each point on image
        for coord in coords:
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(image_rgb, (x_coord, y_coord), 4, (255, 0, 0), 2)
            new_csv_row.extend([x_coord, y_coord])

    cv2.imshow('res', image_rgb)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
