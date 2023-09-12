'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import scipy.io as sio
import numpy as np
import sys
import cv2
import math
import os
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate, \
    Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras import objectives
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import slice

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

batch_size = 2
epochs = 100
img_rows, img_cols = 224, 224
# img_rows, img_cols = 240, 320

x_all_self_v = np.load('./Dataset_rgb_self.npy') / 255.0
x_all_self_t = np.load('./Dataset_rgb_self.npy') / 255.0
# x_all_self = np.concatenate((x_all_self_v, x_all_self_t), axis=3)

x_all_pos_v = np.load('./Dataset_rgb_self.npy') / 255.0
x_all_pos_t = np.load('./Dataset_rgb_self.npy') / 255.0
# x_all_pos = np.concatenate((x_all_pos_v, x_all_pos_t), axis=3)

x_all = np.concatenate((x_all_self_v, x_all_self_t, x_all_pos_v, x_all_pos_t), axis=3)
y_all = np.load('./action.npy')

x_all, y_all = shuffle(x_all, y_all)
x_all = x_all.reshape(x_all.shape[0], img_rows, img_cols, 12)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

image_shape = (img_rows, img_cols, 12)
image_now_next = Input(shape=image_shape, name='image_now_next')
encoderx = BatchNormalization()(image_now_next)
encoderx = Conv2D(32, kernel_size=(11, 11), padding='same', activation='relu', kernel_initializer='he_normal')(encoderx)
encoderx = MaxPooling2D(pool_size=(2, 2))(encoderx)
encoderx = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(encoderx)
encoderx = MaxPooling2D(pool_size=(2, 2))(encoderx)
encoderx = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(encoderx)
encoderx = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(encoderx)
encoderx = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(encoderx)
encoderx = MaxPooling2D(pool_size=(2, 2))(encoderx)
encoderx = Flatten()(encoderx)
latent = Dense(256, activation="relu")(encoderx)
latent_now = Lambda(lambda latent: latent[:, 0:128])(latent)
latent_next = Lambda(lambda latent: latent[:, 128:256])(latent)

action_shape = (3,)
action = Input(shape=action_shape, name='action')
condition = concatenate([latent_now, action])
latent_next_pre = Dense(128, activation="relu")(condition)
latent_next_pre = Dense(128, activation="relu")(latent_next_pre)

# encoderx_now = Lambda(lambda encoderx: encoderx[:, :, :, 0:64])(encoderx)
# encoderx_next = Lambda(lambda encoderx: encoderx[:, :, :, 64:128])(encoderx)
# encoderx_now_flat = Flatten()(encoderx_now)
# encoderx_next_flat = Flatten()(encoderx_next)
# action_shape = (3,)
# action = Input(shape=action_shape, name='action')
# condition = concatenate([encoderx_now_flat, action])
# latent_next = Dense(128, activation="relu")(condition)
# latent_next = Dense(128, activation="relu")(latent_next)

back = Dense(64 * 28 * 28, activation="relu")(latent_next_pre)
back = Reshape((28, 28, 64))(back)
x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(
    back)
x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
x = Conv2DTranspose(32, (11, 11), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
x = Conv2D(6, (1, 1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
image_next_pre = Reshape(target_shape=(img_rows, img_cols, 6))(x)

cvae = Model([image_now_next, action], image_next_pre, name='cvae')

# Calculate custom loss in separate function
def loss_cvae(y_true, y_pred):
    loss_reconstruction = objectives.mse(K.flatten(y_true), K.flatten(y_pred))

    neg_dists = K.sum(K.square(K.l2_normalize(latent_next_pre, axis=1) - K.l2_normalize(latent_now, axis=1)), axis=1, keepdims=True)
    pos_dists = K.sum(K.square(K.l2_normalize(latent_next_pre, axis=1) - K.l2_normalize(latent_next, axis=1)), axis=1, keepdims=True)
    dists = [neg_dists, pos_dists]  # b x b+1
    dists = dists - K.log(tf.math.reduce_sum(K.exp(dists)))
    # dists = tf.nn.log_softmax(dists)
    loss_cpc = K.mean(-dists[:, -1])  # Get last column with is the true pos sample

    loss_all = 0.005 * loss_cpc + loss_reconstruction

    return loss_all

def loss_cp(y_true, y_pred):
    neg_dists = K.sum(K.square(K.l2_normalize(latent_next_pre, axis=1) - K.l2_normalize(latent_now, axis=1)), axis=1, keepdims=True)
    pos_dists = K.sum(K.square(K.l2_normalize(latent_next_pre, axis=1) - K.l2_normalize(latent_next, axis=1)), axis=1, keepdims=True)
    dists = [neg_dists, pos_dists]  # b x b+1
    dists = dists - K.log(tf.math.reduce_sum(K.exp(dists)))
    # dists = tf.nn.log_softmax(dists)
    loss_cpc = K.mean(-dists[:, -1])  # Get last column with is the true pos sample

    return loss_cpc

def loss_re(y_true, y_pred):
    loss_reconstruction = objectives.mse(K.flatten(y_true), K.flatten(y_pred))

    return loss_reconstruction


keras.optimizers.Adam(lr=0.0001, decay=5 * 1e-4)
cvae.compile(loss=loss_cvae,
             optimizer=keras.optimizers.Adam(),
             metrics = [loss_cp, loss_re])
cvae.summary()

# train_history = cvae.fit([x_all, y_all], x_all[:, :, :, 6:12],
#                          batch_size=batch_size,
#                          epochs=epochs,
#                          verbose=1,
#                          shuffle=False,
#                          # validation_split=0.05,
#                          validation_data=([x_all, y_all], x_all[:, :, :, 6:12])
#                          )
#
# # now save...
# # serialize model to JSON
# model_json = cvae.to_json()
# with open("./b0/model_b0.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# cvae.save_weights("./b0/model_b0.h5")
# print("Saved model to disk")
# sys.exit(0)

########################################################################
# # later...
# # load json and create model
json_file = open('./b0/model_b0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./b0/model_b0.h5")
print("Loaded model from disk")

# img = cv2.imread('/home/fanfan/Desktop/point2/tactile017/tactile014' '.png', -1)
# # img = img[:, 330:1050]
# # img = cv2.resize(img, (224, 224))
# x_test = img / 255.0
# x_test = x_test.reshape(1, img_rows, img_cols, 3)
# # x_test = x_test.astype('float32')

x_test = x_all[0, :]
x_test = x_test.reshape(1, img_rows, img_cols, 12)
y_test = y_all[0, :]
y_test = y_test.reshape(1, 3)
image_pre = loaded_model.predict([x_test, y_test])
image_pre = np.reshape(image_pre, newshape=(img_rows, img_cols, 6))
image_pre_rgb = image_pre[:, :, 0:3]
fig, ax = plt.subplots()
ax.imshow(image_pre_rgb)
plt.show()
