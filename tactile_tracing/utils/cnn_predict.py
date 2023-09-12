#!/usr/bin/python

from __future__ import print_function
import sys

sys.path.append('/home/fanfan/z_demo/utils')
import numpy as np
import cv2
import os
import math
import heat_map
from keras.models import model_from_json
from keras import backend as K
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def cnn_predict_b1(self):
    img_rows, img_cols = 224, 224
    rgb_array = []
    tactile_array = []

    image = self.rgb_array[:, 330:1050]
    image = cv2.resize(image, (224, 224))  # right
    rgb_array.append(image)
    Dataset_rgb = np.array(rgb_array)

    image = self.tactile_array[:, 40:280]
    image = cv2.resize(image, (224, 224))
    tactile_array.append(image)
    Dataset_tac = np.array(tactile_array)

    x_all_self = np.concatenate((Dataset_rgb, Dataset_tac), axis=3)
    x_all_self = x_all_self.reshape(x_all_self.shape[0], img_rows, img_cols, 6)
    x_all_self = x_all_self / 255.0
    x_all_self = np.concatenate((x_all_self, x_all_self, x_all_self, x_all_self, x_all_self), axis=0)
    predictions_y = self.loaded_model.predict(x_all_self)
    predictions = predictions_y[0:]

    return predictions


def cnn_predict_b2(self):
    img_rows, img_cols = 224, 224
    rgb_array = []
    tactile_array = []

    image = self.rgb_array[:, 330:1050]
    image = cv2.resize(image, (224, 224))  # right
    rgb_array.append(image)
    Dataset_rgb = np.array(rgb_array)

    image = self.tactile_array[:, 40:280]
    image = cv2.resize(image, (224, 224))
    tactile_array.append(image)
    Dataset_tac = np.array(tactile_array)

    x_all_self = Dataset_tac
    x_all_self = x_all_self.reshape(x_all_self.shape[0], img_rows, img_cols, 3)
    x_all_self = x_all_self / 255.0
    x_all_self = np.concatenate((x_all_self, x_all_self, x_all_self, x_all_self, x_all_self), axis=0)
    predictions_y = self.loaded_model.predict(x_all_self)
    predictions = predictions_y[0:]

    return predictions_y


def cnn_predict_b4(self):
    img_rows, img_cols = 224, 224
    rgb_array = []
    tactile_array = []

    image = self.rgb_array[:, 330:1050]
    image = cv2.resize(image, (224, 224))  # right
    rgb_array.append(image)
    Dataset_rgb = np.array(rgb_array)

    image = self.tactile_array[:, 40:280]
    image = cv2.resize(image, (224, 224))
    tactile_array.append(image)
    Dataset_tac = np.array(tactile_array)

    x_all_self = np.concatenate((Dataset_rgb, Dataset_tac), axis=3)
    x_all_self = Dataset_tac
    x_all_self = x_all_self.reshape(x_all_self.shape[0], img_rows, img_cols, 3)
    x_all_self = x_all_self / 255.0
    # x_all_self = np.concatenate((x_all_self, x_all_self, x_all_self, x_all_self, x_all_self), axis=0)
    # predictions_y = self.loaded_model.predict(x_all_self)
    # predictions = predictions_y[0:]
    predictions = self.loaded_model.predict(x_all_self)

    return predictions
