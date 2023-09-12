# !/usr/bin/python

from __future__ import print_function, division

import math
import tf
import sys
import cv2
import tf.transformations
import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import (
    Point,
    WrenchStamped,
    Point32
)


class PrettyFloat(float):
    def __repr__(self):
        return "%0.2f" % self


def get_depth_real_firstarm(self, data_pc):
    self.clicked_coordinates = (int(self.input_pix[1]), 720 - int(self.input_pix[0]))
    if self.clicked_coordinates:
        # print(self.clicked_coordinates)
        try:
            p = pc2.read_points(data_pc, field_names=("x", "y", "z"), skip_nans=False,
                                uvs=[self.clicked_coordinates]).next()
            clicked_point = PointStamped()
            clicked_point.header = data_pc.header
            clicked_point.point = Point(x=p[0], y=p[1], z=p[2])

            try:
                # print(self.clicked_coordinates)
                point_kinect = self.listener.transformPoint('/camera_2_link', clicked_point)
                if (math.isnan(point_kinect.point.x) or math.isnan(point_kinect.point.y)
                        or math.isnan(point_kinect.point.z)):
                    # print("Could not transform", self.clicked_coordinates)
                    return

                return point_kinect

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                print(tf_err)
        finally:
            self.clicked_coordinates = None


def get_depth_real_secondarm(self, data_pc):
    self.clicked_coordinates = (1280 - int(self.input_pix[1]), int(self.input_pix[0]))
    if self.clicked_coordinates:
        try:
            p = pc2.read_points(data_pc, field_names=("x", "y", "z"), skip_nans=False,
                                uvs=[self.clicked_coordinates]).next()
            clicked_point = PointStamped()
            clicked_point.header = data_pc.header
            clicked_point.point = Point(x=p[0], y=p[1], z=p[2])

            try:
                print(self.clicked_coordinates)
                point_kinect = self.listener.transformPoint('/camera_2_link', clicked_point)
                print(point_kinect)
                if (math.isnan(point_kinect.point.x) or math.isnan(point_kinect.point.y)
                        or math.isnan(point_kinect.point.z)):
                    # print("Could not transform", self.clicked_coordinates)
                    return

                return point_kinect

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                print(tf_err)
        finally:
            self.clicked_coordinates = None


def get_normal_1(self, data_pc):
    self.clicked_coordinates = (int(self.input_pix[0, 0]), int(self.input_pix[0, 1]))
    if self.clicked_coordinates:
        # print(self.clicked_coordinates)
        try:
            # print(self.clicked_coordinates)
            p = pc2.read_points(data_pc, field_names=("x", "y", "z"), skip_nans=False,
                                uvs=[self.clicked_coordinates]).next()
            clicked_point = PointStamped()
            clicked_point.header = data_pc.header
            clicked_point.point = Point(x=p[0], y=p[1], z=p[2])

            try:
                # print(self.clicked_coordinates)
                point_kinect = self.listener.transformPoint('/camera_1_link', clicked_point)
                if (math.isnan(point_kinect.point.x) or math.isnan(point_kinect.point.y)
                        or math.isnan(point_kinect.point.z)):
                    # print("Could not transform", self.clicked_coordinates)
                    return

                return point_kinect

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                print(tf_err)
        finally:
            self.clicked_coordinates = None

def get_normal_2(self, data_pc):
    self.clicked_coordinates = (int(self.input_pix[0, 0]), int(self.input_pix[0, 1]))
    if self.clicked_coordinates:
        # print(self.clicked_coordinates)
        try:
            # print(self.clicked_coordinates)
            p = pc2.read_points(data_pc, field_names=("x", "y", "z"), skip_nans=False,
                                uvs=[self.clicked_coordinates]).next()
            clicked_point = PointStamped()
            clicked_point.header = data_pc.header
            clicked_point.point = Point(x=p[0], y=p[1], z=p[2])

            try:
                # print(self.clicked_coordinates)
                point_kinect = self.listener.transformPoint('/camera_2_link', clicked_point)
                if (math.isnan(point_kinect.point.x) or math.isnan(point_kinect.point.y)
                        or math.isnan(point_kinect.point.z)):
                    # print("Could not transform", self.clicked_coordinates)
                    return

                return point_kinect

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                print(tf_err)
        finally:
            self.clicked_coordinates = None
