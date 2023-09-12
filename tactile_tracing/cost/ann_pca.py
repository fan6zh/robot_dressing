# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import sys
import cv2
import torch
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import torch.nn as nn
import matplotlib.pyplot as plt
import pytouch.tasks.surface3d.geometry as geometry
from pytouch.models import Pix2PixModel, PyTouchZoo


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle


def diffcom(target, base):
    diff = (target * 1.0 - base) / 255 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
    diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
    return diff_abs


def smoothcom(target):
    kernel = np.ones((64, 64), np.float32)
    kernel /= kernel.sum()
    diff_blur = cv2.filter2D(target, -1, kernel)
    return diff_blur


def contourscom(target):
    mask = ((np.abs(target) > 0.02) * 255).astype(np.uint8)
    kernel = np.ones((16, 16), np.uint8)
    mask = cv2.erode(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def visualize_surface_3d():
    src1 = cv2.imread('/home/fanfan/z_tactile/digit/base.png')
    # img = cv2.resize(src1, (160, 120), interpolation=cv2.INTER_AREA)
    # img = img.reshape(3, 320, 240)
    # img = (torch.from_numpy(img) / 255.0 - 0.5) / 0.5
    # output1 = surface3d.point_cloud_3d(img_color=img)
    # output1 = output1.detach().cpu().numpy()

    while True:
        src2 = cv2.imread('/home/fanfan/z_tactile/digit/new.png')
        # img = cv2.resize(src2, (160, 120), interpolation=cv2.INTER_AREA)
        # img = img.reshape(3, 320, 240)
        # img = (torch.from_numpy(img) / 255.0 - 0.5) / 0.5
        # output2 = surface3d.point_cloud_3d(img_color=img)
        # output2 = output2.detach().cpu().numpy()

        # diff = diffcom(output2, output1)
        diff = diffcom(src2, src1)
        diff = smoothcom(diff)
        contours = contourscom(diff)

        # print(contours)
        # cv2.imshow('output', diff)
        # k = cv2.waitKey(1) & 0xFF

        ####################################################### pca
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            print(area)
            # if area < 1e3 or area > 9e3:
            #     continue
            cv2.drawContours(src2, contours, i, (0, 0, 255), 2)
            getOrientation(c, src2)

        cv2.imshow('output', src2)
        k = cv2.waitKey(1) & 0xFF


if __name__ == "__main__":
    visualize_surface_3d()
