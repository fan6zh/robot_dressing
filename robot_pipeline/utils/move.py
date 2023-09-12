# !/usr/bin/python

import numpy as np
import rospy
import math
import baxter_interface

from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import SEAJointState, EndpointState


def move_joint_left(self, data):
    self.startpose = np.array(
        [data.actual_position[0], data.actual_position[1], data.actual_position[2], data.actual_position[3],
         data.actual_position[4], data.actual_position[5], data.actual_position[6]])

    self.distancepose = self.desiredpose - self.startpose
    self.jointVelocities = self.distancepose / self.time_t

    self.angles_left['left_s0'] = self.jointVelocities[0]
    self.angles_left['left_s1'] = self.jointVelocities[1]
    self.angles_left['left_e0'] = self.jointVelocities[2]
    self.angles_left['left_e1'] = self.jointVelocities[3]
    self.angles_left['left_w0'] = self.jointVelocities[4]
    self.angles_left['left_w1'] = self.jointVelocities[5]
    self.angles_left['left_w2'] = self.jointVelocities[6]

    self.limb_left.set_joint_velocities(self.angles_left)


def move_joint_right(self, data):
    self.startpose = np.array(
        [data.actual_position[0], data.actual_position[1], data.actual_position[2], data.actual_position[3],
         data.actual_position[4], data.actual_position[5], data.actual_position[6]])

    self.distancepose = self.desiredpose - self.startpose
    self.jointVelocities = self.distancepose / self.time_t

    self.angles_right['right_s0'] = self.jointVelocities[0]
    self.angles_right['right_s1'] = self.jointVelocities[1]
    self.angles_right['right_e0'] = self.jointVelocities[2]
    self.angles_right['right_e1'] = self.jointVelocities[3]
    self.angles_right['right_w0'] = self.jointVelocities[4]
    self.angles_right['right_w1'] = self.jointVelocities[5]
    self.angles_right['right_w2'] = self.jointVelocities[6]

    self.limb_right.set_joint_velocities(self.angles_right)


def move_joint_both(self, data_left, data_right):
    self.startpose_left = np.array(
        [data_left.actual_position[0], data_left.actual_position[1], data_left.actual_position[2],
         data_left.actual_position[3], data_left.actual_position[4], data_left.actual_position[5],
         data_left.actual_position[6]])
    self.distancepose_left = self.desiredpose_left - self.startpose_left
    self.jointVelocities_left = self.distancepose_left / self.time_t
    self.angles_left['left_s0'] = self.jointVelocities_left[0]
    self.angles_left['left_s1'] = self.jointVelocities_left[1]
    self.angles_left['left_e0'] = self.jointVelocities_left[2]
    self.angles_left['left_e1'] = self.jointVelocities_left[3]
    self.angles_left['left_w0'] = self.jointVelocities_left[4]
    self.angles_left['left_w1'] = self.jointVelocities_left[5]
    self.angles_left['left_w2'] = self.jointVelocities_left[6]

    self.startpose_right = np.array(
        [data_right.actual_position[0], data_right.actual_position[1], data_right.actual_position[2],
         data_right.actual_position[3], data_right.actual_position[4], data_right.actual_position[5],
         data_right.actual_position[6]])
    self.distancepose_right = self.desiredpose_right - self.startpose_right
    self.jointVelocities_right = self.distancepose_right / self.time_t
    self.angles_right['right_s0'] = self.jointVelocities_right[0]
    self.angles_right['right_s1'] = self.jointVelocities_right[1]
    self.angles_right['right_e0'] = self.jointVelocities_right[2]
    self.angles_right['right_e1'] = self.jointVelocities_right[3]
    self.angles_right['right_w0'] = self.jointVelocities_right[4]
    self.angles_right['right_w1'] = self.jointVelocities_right[5]
    self.angles_right['right_w2'] = self.jointVelocities_right[6]

    self.limb_left.set_joint_velocities(self.angles_left)
    self.limb_right.set_joint_velocities(self.angles_right)


def move_end_left(self, data):
    self.jacobianInv = self.kin_left.jacobian_pseudo_inverse()

    self.startpose = np.array(
        [self.grp_curx_le, self.grp_cury_le, self.grp_curz_le, 0, 0, 0, 0])

    self.distancepose = self.desiredpose - self.startpose
    self.desiredEndEffectorVelocities_x = self.distancepose[0] / self.time_t
    self.desiredEndEffectorVelocities_y = self.distancepose[1] / self.time_t
    self.desiredEndEffectorVelocities_z = self.distancepose[2] / self.time_t
    self.desiredEndEffectorVelocities_ax = self.distancepose[3] / self.time_t
    self.desiredEndEffectorVelocities_ay = self.distancepose[4] / self.time_t
    self.desiredEndEffectorVelocities_az = -self.distancepose[5] / self.time_t

    self.desiredEndEffectorVelocitiessecond = np.array(
        [[self.desiredEndEffectorVelocities_x], [self.desiredEndEffectorVelocities_y],
         [self.desiredEndEffectorVelocities_z],
         [self.desiredEndEffectorVelocities_ax], [self.desiredEndEffectorVelocities_ay],
         [self.desiredEndEffectorVelocities_az]])

    self.jointVelocitiessecond = self.jacobianInv * self.desiredEndEffectorVelocitiessecond
    self.jointVelocities = self.jointVelocitiessecond

    self.angles_left['left_s0'] = self.jointVelocities[0]
    self.angles_left['left_s1'] = self.jointVelocities[1]
    self.angles_left['left_e0'] = self.jointVelocities[2]
    self.angles_left['left_e1'] = self.jointVelocities[3]
    self.angles_left['left_w0'] = self.jointVelocities[4]
    self.angles_left['left_w1'] = self.jointVelocities[5]
    self.angles_left['left_w2'] = self.jointVelocities[6]

    self.limb_left.set_joint_velocities(self.angles_left)


def move_end_left_slide(self, data):
    self.jacobianInv = self.kin_left.jacobian_pseudo_inverse()

    self.desiredEndEffectorVelocities_x = self.distancepose[0] / self.time_t
    self.desiredEndEffectorVelocities_y = self.distancepose[1] / self.time_t
    self.desiredEndEffectorVelocities_z = self.distancepose[2] / self.time_t
    self.desiredEndEffectorVelocities_ax = self.distancepose[3] / self.time_t
    self.desiredEndEffectorVelocities_ay = self.distancepose[4] / self.time_t
    self.desiredEndEffectorVelocities_az = -self.distancepose[5] / self.time_t

    self.desiredEndEffectorVelocitiessecond = np.array(
        [[self.desiredEndEffectorVelocities_x], [self.desiredEndEffectorVelocities_y],
         [self.desiredEndEffectorVelocities_z],
         [self.desiredEndEffectorVelocities_ax], [self.desiredEndEffectorVelocities_ay],
         [self.desiredEndEffectorVelocities_az]])

    self.jointVelocitiessecond = self.jacobianInv * self.desiredEndEffectorVelocitiessecond
    self.jointVelocities = self.jointVelocitiessecond

    self.angles_left['left_s0'] = self.jointVelocities[0]
    self.angles_left['left_s1'] = self.jointVelocities[1]
    self.angles_left['left_e0'] = self.jointVelocities[2]
    self.angles_left['left_e1'] = self.jointVelocities[3]
    self.angles_left['left_w0'] = self.jointVelocities[4]
    self.angles_left['left_w1'] = self.jointVelocities[5]
    self.angles_left['left_w2'] = self.jointVelocities[6]

    self.limb_left.set_joint_velocities(self.angles_left)


def move_end_right(self, data):
    self.jacobianInv = self.kin_right.jacobian_pseudo_inverse()

    self.startpose = np.array(
        [self.grp_curx_ri, self.grp_cury_ri, self.grp_curz_ri, 0, 0, 0, 0])

    self.distancepose = self.desiredpose - self.startpose
    self.desiredEndEffectorVelocities_x = self.distancepose[0] / self.time_t
    self.desiredEndEffectorVelocities_y = self.distancepose[1] / self.time_t
    self.desiredEndEffectorVelocities_z = self.distancepose[2] / self.time_t
    self.desiredEndEffectorVelocities_ax = self.distancepose[3] / self.time_t
    self.desiredEndEffectorVelocities_ay = self.distancepose[4] / self.time_t

    self.desiredEndEffectorVelocities_az = self.distancepose[5] / self.time_t

    self.desiredEndEffectorVelocitiessecond = np.array(
        [[self.desiredEndEffectorVelocities_x], [self.desiredEndEffectorVelocities_y],
         [self.desiredEndEffectorVelocities_z],
         [self.desiredEndEffectorVelocities_ax], [self.desiredEndEffectorVelocities_ay],
         [self.desiredEndEffectorVelocities_az]])

    self.jointVelocitiessecond = self.jacobianInv * self.desiredEndEffectorVelocitiessecond
    self.jointVelocities = self.jointVelocitiessecond

    self.angles_right['right_s0'] = self.jointVelocities[0]
    self.angles_right['right_s1'] = self.jointVelocities[1]
    self.angles_right['right_e0'] = self.jointVelocities[2]
    self.angles_right['right_e1'] = self.jointVelocities[3]
    self.angles_right['right_w0'] = self.jointVelocities[4]
    self.angles_right['right_w1'] = self.jointVelocities[5]
    self.angles_right['right_w2'] = self.jointVelocities[6]

    self.limb_right.set_joint_velocities(self.angles_right)


def move_rotate_left(self, data):
    self.jacobianInv = self.kin_left.jacobian_pseudo_inverse()

    self.startpose = np.array(
        [self.grp_curx_le, self.grp_cury_le, self.grp_curz_le, 0, 0, self.euler[2], 0])

    self.distancepose = self.desiredpose - self.startpose
    self.desiredEndEffectorVelocities_x = self.distancepose[0] / self.time_t
    self.desiredEndEffectorVelocities_y = self.distancepose[1] / self.time_t
    self.desiredEndEffectorVelocities_z = self.distancepose[2] / self.time_t
    self.desiredEndEffectorVelocities_ax = self.distancepose[3] / self.time_t
    self.desiredEndEffectorVelocities_ay = self.distancepose[4] / self.time_t
    self.desiredEndEffectorVelocities_az = self.distancepose[5] / self.time_t

    self.desiredEndEffectorVelocitiessecond = np.array(
        [[self.desiredEndEffectorVelocities_x], [self.desiredEndEffectorVelocities_y],
         [self.desiredEndEffectorVelocities_z],
         [self.desiredEndEffectorVelocities_ax], [self.desiredEndEffectorVelocities_ay],
         [self.desiredEndEffectorVelocities_az]])

    self.jointVelocitiessecond = self.jacobianInv * self.desiredEndEffectorVelocitiessecond
    self.jointVelocities = self.jointVelocitiessecond

    self.angles_left['left_s0'] = self.jointVelocities[0]
    self.angles_left['left_s1'] = self.jointVelocities[1]
    self.angles_left['left_e0'] = self.jointVelocities[2]
    self.angles_left['left_e1'] = self.jointVelocities[3]
    self.angles_left['left_w0'] = self.jointVelocities[4]
    self.angles_left['left_w1'] = self.jointVelocities[5]
    self.angles_left['left_w2'] = self.jointVelocities[6]

    self.limb_left.set_joint_velocities(self.angles_left)


def move_rotate_right(self, data):
    self.jacobianInv = self.kin_right.jacobian_pseudo_inverse()

    self.startpose = np.array(
        [self.grp_curx_ri, self.grp_cury_ri, self.grp_curz_ri, 0, 0, self.euler[2], 0])

    self.distancepose = self.desiredpose - self.startpose
    self.desiredEndEffectorVelocities_x = self.distancepose[0] / self.time_t
    self.desiredEndEffectorVelocities_y = self.distancepose[1] / self.time_t
    self.desiredEndEffectorVelocities_z = self.distancepose[2] / self.time_t
    self.desiredEndEffectorVelocities_ax = self.distancepose[3] / self.time_t
    self.desiredEndEffectorVelocities_ay = self.distancepose[4] / self.time_t
    self.desiredEndEffectorVelocities_az = self.distancepose[5] / self.time_t

    self.desiredEndEffectorVelocitiessecond = np.array(
        [[self.desiredEndEffectorVelocities_x], [self.desiredEndEffectorVelocities_y],
         [self.desiredEndEffectorVelocities_z],
         [self.desiredEndEffectorVelocities_ax], [self.desiredEndEffectorVelocities_ay],
         [self.desiredEndEffectorVelocities_az]])

    self.jointVelocitiessecond = self.jacobianInv * self.desiredEndEffectorVelocitiessecond
    self.jointVelocities = self.jointVelocitiessecond

    self.angles_right['right_s0'] = self.jointVelocities[0]
    self.angles_right['right_s1'] = self.jointVelocities[1]
    self.angles_right['right_e0'] = self.jointVelocities[2]
    self.angles_right['right_e1'] = self.jointVelocities[3]
    self.angles_right['right_w0'] = self.jointVelocities[4]
    self.angles_right['right_w1'] = self.jointVelocities[5]
    self.angles_right['right_w2'] = self.jointVelocities[6]

    self.limb_right.set_joint_velocities(self.angles_right)
