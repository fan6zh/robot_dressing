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


def cnn_predict_1(self):
    # K.clear_session()

    image_l = cv2.imread('./maskrcnn/gown_val/image_1.png', -1)
    image = image_l[:, 280:1000] / 255.0
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1, 224, 224, 3)

    # later...
    # load json and create model
    json_file = open('./model_1/model_edge.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model_1/model_edge.h5")
    print("Loaded model from disk")

    im_pre = loaded_model.predict(image)
    im_pre = im_pre.reshape(224, 224, 1)
    im_pre = np.concatenate((im_pre, im_pre, im_pre), axis=2)
    im_pre = cv2.resize(im_pre, (720, 720))

    #fig, ax = plt.subplots()
    #ax.imshow(im_pre)
    #plt.show()

    pt_lt = []
    for i in range(int(self.pixel.y) + 55, int(self.pixel.y) + 65):
        for j in range(int(self.pixel.x) - 10 - 280, int(self.pixel.x) + 10 - 280):
            if im_pre[i, j][0] > 0.5:
                pt_lt.append([i, j + 280])

    print(len(pt_lt))
    ind = np.random.randint(len(pt_lt))
    self.point = pt_lt[ind]

    image_pre = cv2.circle(image_l, (self.point[1], self.point[0]), 5, (0, 255, 0), -1)
    cv2.imwrite("./image/image_1rgb_pre.png", image_pre)


def cnn_predict_2(self):
    img_rows, img_cols = 224, 224
    array_X = []

    image = self.image[:, 330:1050]
    image = cv2.resize(image, (224, 224))  # right
    array_X.append(image)
    Dataset_X = np.array(array_X)
    x_test = Dataset_X / 255.0
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    predictions_y = self.loaded_model.predict(x_test)

    return predictions_y


def cnn_predict_4(self):
    img_rows, img_cols = 224, 224
    array_X = []

    image_l = cv2.imread('./image/image_4rgb.png', -1)
    image = image_l[:, 120:840]  # inverse
    image = cv2.resize(image, (224, 224))  # right

    array_X.append(image)
    Dataset_X = np.array(array_X)
    x_test = Dataset_X / 255.0
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    # later...
    # load json and create model
    json_file = open('./model_4/model_heatmap.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model_4/model_heatmap.h5")
    print("Loaded model from disk")

    predictions_y = loaded_model.predict(x_test)
    heatmap = predictions_y
    heatmap = np.reshape(heatmap, newshape=(224, 224))  # inverse
    # heat_map.add(image, heatmap, alpha=0.5)
    indices = np.where(heatmap == heatmap.max())
    self.predictions = np.array([[indices[1][0], indices[0][0]]])
    self.predictions[0, 0] = math.ceil(self.predictions[0, 0] * 720 / 224 + 120)
    self.predictions[0, 1] = math.ceil(self.predictions[0, 1] * 720 / 224)
    print(self.predictions)

    image_pre = cv2.circle(image_l, (self.predictions[0, 0], self.predictions[0, 1]), 5, (0, 255, 0), -1)
    cv2.imwrite("./image/image_4rgb_pre.png", image_pre)


def cnn_predict_5_corner(self):
    img_rows, img_cols = 224, 224
    array_X = []

    image_l = cv2.imread('./image/image_5rgb_corner.png', -1)
    image = image_l[:, 420:1140]
    image = cv2.resize(image, (224, 224))

    array_X.append(image)
    Dataset_X = np.array(array_X)
    x_test = Dataset_X / 255.0
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    # later...
    # load json and create model
    json_file = open('./model_5/model_heatmap_corner.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model_5/model_heatmap_corner.h5")
    print("Loaded model from disk")

    predictions_y = loaded_model.predict(x_test)
    heatmap = predictions_y / 100
    heatmap = np.reshape(heatmap, newshape=(224, 224))
    indices = np.where(heatmap == heatmap.max())
    self.predictions = np.array([[indices[1][0], indices[0][0]]])

    self.predictions[0, 0] = math.ceil(self.predictions[0, 0] * 720 / 224 + 420 - 5)
    self.predictions[0, 1] = math.ceil(self.predictions[0, 1] * 720 / 224)
    print(self.predictions)

    image_pre = cv2.circle(image_l, (self.predictions[0, 0], self.predictions[0, 1]), 5, (0, 255, 0), -1)
    cv2.imwrite("./image/image_5rgb_corer_pre.png", image_pre)


def cnn_predict_5_grasp(self):
    img_rows, img_cols = 224, 224
    array_X = []

    image_l = cv2.imread('./image/image_5rgb_grasp.png', -1)
    image = image_l[:, 420:1140]
    image = cv2.resize(image, (224, 224))

    array_X.append(image)
    Dataset_X = np.array(array_X)
    x_test = Dataset_X / 255.0
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    # later...
    # load json and create model
    json_file = open('./model_5/model_heatmap_grasp.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model_5/model_heatmap_grasp.h5")
    print("Loaded model from disk")

    predictions_y = loaded_model.predict(x_test)
    heatmap = predictions_y
    heatmap = np.reshape(heatmap, newshape=(224, 224))
    indices = np.where(heatmap == heatmap.max())
    print(heatmap.max())
    self.predictions = np.array([[indices[1][0], indices[0][0]]])
    print(self.predictions)

    self.predictions[0, 0] = math.ceil(self.predictions[0, 0] * 720 / 224 + 420)
    self.predictions[0, 1] = math.ceil(self.predictions[0, 1] * 720 / 224)

    image_pre = cv2.circle(image_l, (self.predictions[0, 0], self.predictions[0, 1]), 5, (0, 255, 0), -1)
    cv2.imwrite("./image/image_5rgb_grasp_pre.png", image_pre)
