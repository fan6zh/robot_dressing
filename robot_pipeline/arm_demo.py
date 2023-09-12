#!/usr/bin/python

from __future__ import print_function, division

import sys

sys.dont_write_bytecode = True
sys.path.append('/home/fanfan/z_demo/utils')
import hand
# import dekr
import hrnet
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
import move
import baxter_interface

from scipy import ndimage
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

        self.listen00 = False
        self.listen0 = False
        self.listen1 = False
        self.listen2 = False
        self.listen3 = False
        self.listen4 = False
        self.listen5 = False
        self.listen6 = False
        self.listen7 = False
        self.listen8 = False
        self.listen9 = False
        self.listen10 = False
        self.listen10_0 = False
        self.listen11 = False
        self.listen12 = False
        self.listen13 = False
        self.listen14 = False
        self.listen15 = False
        self.listen16 = False

        self.transf_2 = sio.loadmat('/home/fanfan/Desktop/2022/tf/tf_bax_came_2.mat')
        self.tform_2 = self.transf_2['tf_bax_came_2']

        ##############################################################
        hrnet.hrnet_load(self)

        self.subscriber00 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                             self.callback_00, queue_size=1)

        self.subscriber0 = rospy.Subscriber(rgb_topic, Image, self.callback_0, queue_size=1)
        self.subscriber1 = rospy.Subscriber(pc_topic, PointCloud2, self.callback_1, queue_size=1)

        self.subscriber2 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                            self.callback_2, queue_size=1)
        self.subscriber3 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                            self.callback_3, queue_size=1)
        self.subscriber4 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                            self.callback_4, queue_size=1)

        self.subscriber5 = rospy.Subscriber(rgb_topic, Image, self.callback_5, queue_size=1)
        self.subscriber6 = rospy.Subscriber(pc_topic, PointCloud2, self.callback_6, queue_size=1)

        self.subscriber7 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_7, queue_size=1)
        self.subscriber8 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_8, queue_size=1)
        self.subscriber9 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                            self.callback_9, queue_size=1)
        self.subscriber10 = rospy.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState,
                                             self.callback_10, queue_size=1)
        self.subscriber11 = rospy.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState,
                                             self.callback_11, queue_size=1)

        self.subscriber12 = message_filters.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState)
        self.subscriber13 = message_filters.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState)
        self.tss13 = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber12, self.subscriber13], queue_size=1, slop=0.005)
        self.tss13.registerCallback(self.callback_13)

        self.subscriber14 = message_filters.Subscriber("/robot/limb/left/gravity_compensation_torques", SEAJointState)
        self.subscriber15 = message_filters.Subscriber("/robot/limb/right/gravity_compensation_torques", SEAJointState)
        self.tss15 = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber14, self.subscriber15], queue_size=1, slop=0.005)
        self.tss15.registerCallback(self.callback_15)

        self.subscriber_cur_le = rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState,
                                                  self.callback_cur_le, queue_size=1)
        self.subscriber_cur_ri = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState,
                                                  self.callback_cur_ri, queue_size=1)

        self.start = time.time()

        self.time_t = 2
        self.thresh_joint = 0.08
        self.thresh_end = 0.015
        self.sh = 0.02
        self.l_hand = 0.18
        self.grasp_place_1 = 0.02
        self.d_trajectory = 0.1

        #########################################
        self.listen00 = True
        ##########################################
        # self.rob_shou = np.array([[0.668112],
        #                           [0.605203],
        #                           [-0.389591],
        #                           [1.]])

    def callback_00(self, data):  # up
        if self.listen00:
            self.desiredpose = np.array(
                [-0.3616359707439863, -1.2432914285811278, 0.16336895390979655, 1.8660876284626058,
                 -0.09050486648523941, 0.9648739155799253, 0.6818544602150665])

            move.move_joint_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber00.unregister()
                robotiq.robotiq_left(0)
                print('start tracking')
                self.listen0 = True

    def callback_0(self, data_rgb):
        if self.listen0:
            cv_image = self.bridge.imgmsg_to_cv2(data_rgb, "bgr8")
            self.rotate_cv_image = ndimage.rotate(cv_image, -90)
            hrnet.hrnet_predict(self)
            # dekr.dekr_predict(self)

            self.input_pix_hand_r = self.preds[0][10]
            self.input_pix_elbow_r = self.preds[0][8]
            self.input_pix_shou_r = self.preds[0][6]

            self.input_pix_hand_l = self.preds[0][9]
            self.input_pix_elbow_l = self.preds[0][7]
            self.input_pix_shou_l = self.preds[0][5]

            if self.input_pix_hand_r[0] < 400 and self.input_pix_elbow_r[0] < 400 and self.input_pix_shou_r[0] < 400:
                self.input_pix_hand = self.input_pix_hand_r
                self.input_pix_elbow = self.input_pix_elbow_r
                self.input_pix_shou = self.input_pix_shou_r
            else:
                self.input_pix_hand = self.input_pix_hand_l
                self.input_pix_elbow = self.input_pix_elbow_l
                self.input_pix_shou = self.input_pix_shou_l

            if len(self.preds) > 0 and self.input_pix_elbow[0] - self.input_pix_hand[0] < 70 and self.input_pix_hand[
                0] < 400 and self.input_pix_elbow[0] < 400 and self.input_pix_shou[0] < 400:
                self.subscriber0.unregister()
                self.listen1 = True

    def callback_1(self, data_pc):
        if self.listen1:
            self.input_pix = self.input_pix_hand
            self.point_kinect_hand = get_depth.get_depth_real_firstarm(self, data_pc)
            self.input_pix = self.input_pix_elbow
            self.point_kinect_elbow = get_depth.get_depth_real_firstarm(self, data_pc)
            # self.input_pix = self.input_pix_shou
            # self.point_kinect_shou = get_depth.get_depth_real_firstarm(self, data_pc)

            self.kinect_hand = np.matrix(
                [[self.point_kinect_hand.point.x, self.point_kinect_hand.point.y,
                  self.point_kinect_hand.point.z]])
            self.kinect_elbow = np.matrix(
                [[self.point_kinect_elbow.point.x, self.point_kinect_elbow.point.y,
                  self.point_kinect_elbow.point.z]])
            # self.kinect_shou = np.matrix(
            #     [[self.point_kinect_shou.point.x, self.point_kinect_shou.point.y,
            #       self.point_kinect_shou.point.z]])
            self.rob_hand = self.tform_2 * augment_data(self.kinect_hand).transpose()
            self.rob_elbow = self.tform_2 * augment_data(self.kinect_elbow).transpose()
            # self.rob_shou = self.tform_2 * augment_data(self.kinect_shou).transpose()

            self.l_forward = math.sqrt(math.pow((self.rob_hand[0] - self.rob_elbow[0]), 2) + math.pow(
                (self.rob_hand[1] - self.rob_elbow[1]), 2) + math.pow((self.rob_hand[2] - self.rob_elbow[2]), 2))
            # self.l_back = math.sqrt(math.pow((self.rob_shou[0] - self.rob_elbow[0]), 2) + math.pow(
            #     (self.rob_shou[1] - self.rob_elbow[1]), 2) + math.pow((self.rob_shou[2] - self.rob_elbow[2]), 2))
            self.l_back = self.l_forward + 0.02
            self.l_all = self.l_forward + self.l_back

            self.shx = (self.l_all * self.rob_elbow[0] - self.l_back * self.rob_hand[0]) / self.l_forward
            self.shy = (self.l_all * self.rob_elbow[1] - self.l_back * self.rob_hand[1]) / self.l_forward
            self.shz = (self.l_all * self.rob_elbow[2] - self.l_back * self.rob_hand[2]) / self.l_forward
            self.rob_shou = np.array([[self.shx[0, 0]],
                                      [self.shy[0, 0]],
                                      [self.shz[0, 0]],
                                      [1.]])
            print(self.rob_shou)

            self.graspx = ((self.grasp_place_1 + self.l_forward) * (self.rob_shou[0] - self.rob_hand[0]) +
                           self.l_all * self.rob_hand[0]) / self.l_all
            self.graspy = ((self.grasp_place_1 + self.l_forward) * (self.rob_shou[1] - self.rob_hand[1]) +
                           self.l_all * self.rob_hand[1]) / self.l_all
            self.graspz = ((self.grasp_place_1 + self.l_forward) * (self.rob_shou[2] - self.rob_hand[2]) +
                           self.l_all * self.rob_hand[2]) / self.l_all
            self.angle_grasp = np.arctan(
                abs(self.rob_shou[0] - self.rob_elbow[0]) / abs(self.rob_shou[1] - self.rob_elbow[1]))

            self.subscriber1.unregister()
            cv2.destroyAllWindows()
            self.listen2 = True

    def callback_2(self, data):
        if self.listen2:
            self.euler = tf.transformations.euler_from_quaternion(self.quaternion_le)
            self.desiredpose = np.array(
                [self.graspx[0, 0] + 0.00, self.graspy[0, 0], self.graspz[0, 0] + 0.05,
                 0, 0, math.pi / 2 - self.angle_grasp, 0])

            move.move_rotate_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber2.unregister()
                print('reached')
                self.listen3 = True

    def callback_3(self, data):  # grasp
        if self.listen3:
            self.desiredpose = np.array(
                [self.graspx[0, 0] + 0.00, self.graspy[0, 0], self.graspz[0, 0] - 0.05,
                 0, 0, 0, 0])

            move.move_end_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber3.unregister()
                robotiq.robotiq_left(255)
                print('grasped')
                self.listen4 = True

    def callback_4(self, data):  # lift
        if self.listen4:
            self.desiredpose = np.array(
                [-0.35204859081970247, -0.2741990658345177, 0.0674951546669582, 1.403975916112125, -0.6082233823965666,
                 0.6281651326390769, 2.662607152572107])

            move.move_joint_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber4.unregister()
                print('lifted')
                self.start = time.time()
                self.listen5 = True

    def callback_5(self, data_rgb):
        if self.listen5:
            self.cv_image = self.bridge.imgmsg_to_cv2(data_rgb, "bgr8")
            self.cv_image = ndimage.rotate(self.cv_image, -90)
            self.cv_image = self.cv_image[:, 0:520]
            hand.hand_predict(self)

            if time.time() - self.start > 15 and np.array(self.results).shape[0] == 1:
                self.result = self.results[0]
                id, name, confidence, x, y, w, h = self.result
                self.input_pix_hand = np.array([[x + w / 2], [y]])
                print(self.input_pix_hand)

                self.subscriber5.unregister()
                self.listen6 = True

    def callback_6(self, data_pc):
        if self.listen6:
            self.input_pix = self.input_pix_hand
            self.point_kinect_hand = get_depth.get_depth_real_firstarm(self, data_pc)

            self.kinect_hand = np.matrix(
                [[self.point_kinect_hand.point.x, self.point_kinect_hand.point.y,
                  self.point_kinect_hand.point.z]])
            self.rob_hand = self.tform_2 * augment_data(self.kinect_hand).transpose()
            print(self.rob_hand)

            self.l_all = math.sqrt(math.pow((self.rob_shou[0] - self.rob_hand[0]), 2) + math.pow(
                (self.rob_shou[1] - self.rob_hand[1]), 2) + math.pow((self.rob_shou[2] - self.rob_hand[2]), 2))
            self.update_fingx = self.l_hand * (self.rob_hand[0] - self.rob_shou[0]) / (
                self.l_all) + self.rob_hand[0]
            self.update_fingy = self.l_hand * (self.rob_hand[1] - self.rob_shou[1]) / (
                self.l_all) + self.rob_hand[1]
            self.update_fingz = self.l_hand * (self.rob_hand[2] - self.rob_shou[2]) / (
                self.l_all) + self.rob_hand[2]

            cv2.destroyAllWindows()
            self.subscriber5.unregister()
            self.subscriber6.unregister()
            print('track done')
            self.listen7 = True

    ####################################################################################
    def callback_7(self, data):  # finger
        if self.listen7:
            self.desiredpose = np.array(
                [self.update_fingx[0, 0] + self.sh, self.update_fingy[0, 0],
                 self.update_fingz[0, 0] + self.d_trajectory, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber7.unregister()
                print('finger reached')
                self.listen8 = True

    def callback_8(self, data):  # hand
        if self.listen8:
            self.desiredpose = np.array(
                [self.rob_hand[0, 0] + self.sh, self.rob_hand[1, 0],
                 self.rob_hand[2, 0] + self.d_trajectory, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber8.unregister()
                print('hand reached')
                self.listen9 = True

    def callback_9(self, data):  # middle
        if self.listen9:
            self.desiredpose = np.array(
                [(self.update_fingx[0, 0] + self.rob_shou[0, 0]) / 2 + self.sh,
                 (self.update_fingy[0, 0] + self.rob_shou[1, 0]) / 2,
                 (self.update_fingz[0, 0] + self.rob_shou[2, 0]) / 2 + self.d_trajectory - 0.03, 0, 0, 0, 0])

            move.move_end_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_end:
                self.subscriber9.unregister()
                print('middle reached')
                self.listen10 = True

    def callback_10(self, data):  # shoulder
        if self.listen10:
            self.desiredpose = np.array(
                [1.101398205701727, 0.1004757416064946, 0.36623791310764253, 0.6768690226544388, 0.9828981898375788,
                 0.5579855115933192, -0.40151947122900705])

            move.move_joint_right(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint:
                self.subscriber10.unregister()
                robotiq.robotiq_right(0)
                print('shoulder reached')
                self.listen11 = True

    def callback_11(self, data):  # drop
        if self.listen11:
            self.desiredpose = np.array(
                [-0.6937428113211783, -0.0502378708032473, 0.4157087935169471, 1.1815487018687398, -0.7355437877910559,
                 0.6304661038209051, 2.476995477237972])

            move.move_joint_left(self, data)

            if np.sqrt(np.sum(np.square(self.distancepose))) < self.thresh_joint + 0.08:
                self.subscriber11.unregister()
                print('dropped')
                robotiq.robotiq_left(0)
                self.listen13 = True

    def callback_13(self, data_left, data_right):  # up
        if self.listen13:
            self.desiredpose_left = np.array(
                [-0.5127330783506996, -0.23700003172829642, 0.16528642989465334, 1.39975746894544, -0.49700977527487405,
                 0.44600491407768406, 2.5126605305563077])
            self.desiredpose_right = np.array(
                [1.408577858475781, -0.6066894016086811, -0.15263108839459866, 1.5416506918248407, 0.6703496043059258,
                 0.5890486225479988, -0.0502378708032473])

            move.move_joint_both(self, data_left, data_right)

            if np.sqrt(np.sum(np.square(self.distancepose_right))) < self.thresh_joint:
                self.subscriber12.unregister()
                self.subscriber13.unregister()
                self.listen15 = True

    def callback_15(self, data_left, data_right):  # back
        if self.listen15:
            self.desiredpose_left = np.array(
                [-0.3393932493196478, -1.2233496783386175, 0.13882526130362993, 1.792456550644106, -0.3992185000471789,
                 0.9828981898375788, 0.23354857495555426])
            self.desiredpose_right = np.array(
                [0.3739078170470696, -1.2103108416415915, -0.06059224112147384, 1.7541070309469706,
                 -0.02339320701525256, 1.0151117863831725, -0.4770680250323637])

            move.move_joint_both(self, data_left, data_right)

            if np.sqrt(np.sum(np.square(self.distancepose_right))) < self.thresh_joint and np.sqrt(
                    np.sum(np.square(self.distancepose_left))) < self.thresh_joint:
                self.subscriber14.unregister()
                self.subscriber15.unregister()

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
            if self.listen16:
                break


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    rospy.init_node("arm_1")

    armcontroller = ArmController()
    armcontroller.breakcode()

    # rospy.spin()
