#!/usr/bin/python

from __future__ import print_function, division

import sys

sys.dont_write_bytecode = True
sys.path.append('/home/fanfan/z_demo/utils')
import numpy as np
import scipy.io as sio
import rospy
import math
import time
import message_filters
import tf
import cv2
import get_depth
import serial
import robotiq
import cnn_predict
import move
import baxter_interface

from sensor_msgs.msg import PointCloud2, Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from tf_finder import augment_data
from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import SEAJointState, EndpointState


class ArmController:
    def __init__(self):
        rgb_topic = rospy.get_param("~rgb_topic_1", "/camera_2/color/image_rect_color")
        pc_topic = rospy.get_param("~pc_topic_1", "/camera_2/depth_registered/points")

        self.clicked_coordinates = None
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        self.bridge = CvBridge()

        self.kin_left = baxter_kinematics('left')
        self.limb_left = baxter_interface.Limb('left')
        self.angles_left = self.limb_left.joint_angles()
        self.kin_right = baxter_kinematics('right')
        self.limb_right = baxter_interface.Limb('right')
        self.angles_right = self.limb_right.joint_angles()

        self.listen0 = False
        self.listen1 = False
        self.listen2 = False
        self.listen3 = False
        self.listen4 = False
        self.listen5 = False
        self.listen6 = False
        self.listen60 = False
        self.listen600 = False
        self.listen7 = False
        self.listen8 = False
        self.listen9 = False
        self.listen10 = False
        self.listen11 = False
        self.listen12 = False
        self.listen13 = False

        self.time_t = 2
        self.thresh_joint = 0.08
        self.thresh_end = 0.015

        self.transf_2 = sio.loadmat('/home/fanfan/Desktop/2022/tf/tf_bax_came_2.mat')
        self.tform_2 = self.transf_2['tf_bax_came_2']

        ##############################################################
        self.subscriber00 = message_filters.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState)
        self.subscriber01 = message_filters.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState)
        self.tss15 = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber00, self.subscriber01], queue_size=1, slop=0.005)
        self.tss15.registerCallback(self.callback_0)

        self.subscriber1 = rospy.Subscriber("/camera_2/color/image_rect_color", Image, self.callback_1,
                                            queue_size=1)
        self.subscriber2 = rospy.Subscriber(pc_topic, PointCloud2, self.callback_2, queue_size=1)
        self.subscriber3 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_3, queue_size=1)
        self.subscriber4 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_4, queue_size=1)
        self.subscriber5 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_5, queue_size=1)
        self.subscriber6 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_6, queue_size=1)
        self.subscriber600 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                              self.callback_600, queue_size=1)
        self.subscriber60 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                             self.callback_60, queue_size=1)

        self.subscriber7 = rospy.Subscriber("/camera_2/color/image_rect_color", Image, self.callback_7,
                                            queue_size=1)
        self.subscriber8 = rospy.Subscriber(pc_topic, PointCloud2, self.callback_8, queue_size=1)
        self.subscriber9 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                            self.callback_9, queue_size=1)
        self.subscriber10 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                             self.callback_10, queue_size=1)
        self.subscriber11 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                             self.callback_11, queue_size=1)
        self.subscriber12 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                             self.callback_12, queue_size=1)

        self.subscriber_cur_le = rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState,
                                                  self.callback_cur_le, queue_size=1)
        self.subscriber_cur_ri = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState,
                                                  self.callback_cur_ri, queue_size=1)
        self.start = time.time()

        ##########################################
        self.listen0 = True

    def callback_0(self, data_left, data_right):  # back
        if self.listen0:
            self.desiredpose_left = np.array(
                [-0.6745680514726107, -1.0120438248074017, 0.5365097805629234, 2.0455633806451994, -0.5656554155327463,
                 0.6216457142905639, 0.8536603084582327])
            self.desiredpose_right = np.array(
                [0.30181072001645515, -1.0952622825501854, -0.057524279545703015, 1.6444274046131635,
                 0.41494180312300444, 1.0672671331712766, -0.5698738626994312])

            move.move_joint_both(self, data_left, data_right)

            if np.sqrt(np.sum(np.square(self.distancepose_right))) < self.thresh_joint and np.sqrt(
                    np.sum(np.square(self.distancepose_left))) < self.thresh_joint:
                self.subscriber00.unregister()
                self.subscriber01.unregister()
                robotiq.robotiq_left(0)
                robotiq.robotiq_right(0)
                self.listen1 = True
                #self.listen7 = True

    def callback_1(self, data):
        if self.listen1:
            ######################################################################################## real depth image
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            depth_array = np.array(cv_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            cv2.imwrite("./image/image_5rgb_corner.png", depth_array * 255)
            print("rgb image saved!")

            self.subscriber1.unregister()

            cnn_predict.cnn_predict_5_corner(self)
            self.listen2 = True

    def callback_2(self, data_pc):
        if self.listen2:
            self.input_pix = self.predictions
            self.point = get_depth.get_normal_2(self, data_pc)
            self.camera_point = np.matrix([[self.point.point.x, self.point.point.y, self.point.point.z]])
            self.rob_point = self.tform_2 * augment_data(self.camera_point).transpose()
            print(self.rob_point)

            self.subscriber2.unregister()
            self.listen3 = True

    def callback_3(self, data):
        if self.listen3:
            self.desiredpose = np.array(
                [self.rob_point[0, 0] - 0.025, self.rob_point[1, 0] - 0.02, self.rob_point[2, 0] + 0.03, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber3.unregister()
                self.listen4 = True

    def callback_4(self, data):
        if self.listen4:
            self.desiredpose = np.array(
                [self.rob_point[0, 0] - 0.025, self.rob_point[1, 0] - 0.02, self.rob_point[2, 0] - 0.02, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber4.unregister()
                robotiq.robotiq_right(255)
                self.listen5 = True

    def callback_5(self, data):
        if self.listen5:
            self.desiredpose = np.array(
                [self.rob_point[0, 0], self.rob_point[1, 0], self.rob_point[2, 0] + 0.06, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber5.unregister()
                self.listen6 = True

    def callback_6(self, data):
        if self.listen6:
            self.desiredpose = np.array(
                [0.7715923363063631, -0.41302432713814763, 0.041033986075934815, 1.6701215828102443,
                 0.011888351106111956, 0.18177672336442152, 0.12003399665203363])

            move.move_joint_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber6.unregister()
                robotiq.robotiq_right(0)
                self.listen600 = True

    def callback_600(self, data):
        if self.listen600:
            self.desiredpose = np.array(
                [1.0615147052167062, -0.5894321177449703, -0.184461189743221, 1.734932271098403, 0.32482043183473636,
                 0.4456214188807127, -0.06212622190935926])

            move.move_joint_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber600.unregister()
                robotiq.robotiq_right(0)
                self.listen60 = True

    def callback_60(self, data):
        if self.listen60:
            self.desiredpose = np.array(
                [0.3739078170470696, -1.2103108416415915, -0.06059224112147384, 1.7541070309469706,
                 -0.02339320701525256, 1.0151117863831725, -0.4770680250323637])

            move.move_joint_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber60.unregister()
                self.listen7 = True

    def callback_7(self, data):
        if self.listen7:
            ######################################################################################## real depth image
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            depth_array = np.array(cv_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            cv2.imwrite("./image/image_5rgb_grasp.png", depth_array * 255)
            print("rgb image saved!")

            self.subscriber7.unregister()

            cnn_predict.cnn_predict_5_grasp(self)
            print(self.predictions)
            robotiq.robotiq_left(0)
            robotiq.robotiq_right(0)
            self.listen8 = True

    def callback_8(self, data_pc):
        if self.listen8:
            self.input_pix = self.predictions
            self.point = get_depth.get_normal_2(self, data_pc)
            self.camera_point = np.matrix([[self.point.point.x, self.point.point.y, self.point.point.z]])
            self.rob_point = self.tform_2 * augment_data(self.camera_point).transpose()
            print(self.rob_point)

            self.subscriber8.unregister()
            self.listen9 = True

    def callback_9(self, data):
        if self.listen9:
            self.desiredpose = np.array(
                [self.rob_point[0, 0] - 0.01, self.rob_point[1, 0] + 0.00, self.rob_point[2, 0] + 0.03, 0, 0, 0, 0])

            move.move_end_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber9.unregister()
                self.listen10 = True

    def callback_10(self, data):
        if self.listen10:
            self.desiredpose = np.array(
                [self.rob_point[0, 0] - 0.01, self.rob_point[1, 0] + 0.00, self.rob_point[2, 0] - 0.015, 0, 0, 0, 0])

            move.move_end_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber10.unregister()
                robotiq.robotiq_left(255)
                self.listen11 = True

    def callback_11(self, data):
        if self.listen11:
            self.desiredpose = np.array(
                [self.rob_point[0, 0], self.rob_point[1, 0], self.rob_point[2, 0] + 0.05, 0, 0, 0, 0])

            move.move_end_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber11.unregister()
                self.listen12 = True

    def callback_12(self, data):
        if self.listen12:
            self.desiredpose = np.array(
                [-0.02339320701525256, -0.7286408742455716, 0.09817477042466648, 1.7245779007801765,
                 -0.4920243377142465, 0.8808884674431988, 1.1209564607472662])

            move.move_joint_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber12.unregister()
                self.listen13 = True

    def callback_cur_le(self, data):
        self.grp_curx_le = data.pose.position.x
        self.grp_cury_le = data.pose.position.y
        self.grp_curz_le = data.pose.position.z
        self.quaternion_le = (data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
                              data.pose.orientation.w)

    def callback_cur_ri(self, data):
        self.grp_curx_ri = data.pose.position.x
        self.grp_cury_ri = data.pose.position.y
        self.grp_curz_ri = data.pose.position.z
        self.quaternion_ri = (data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
                              data.pose.orientation.w)

    def breakcode(self):
        while not rospy.is_shutdown():
            if self.listen13:
                break


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    rospy.init_node("clothes_5")

    armcontroller = ArmController()
    armcontroller.breakcode()

    # rospy.spin()
