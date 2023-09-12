#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)
import sys
import os
import time
import math
import numpy as np
import bpy, bpy_extras
import traceback
import copy
import mathutils
import matplotlib.pyplot as plt

from pathlib import Path
from termcolor import colored
from collections import defaultdict
from mathutils import Matrix
from blendtorch import btb
from mathutils import *


class PregraspEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.gripper = bpy.data.objects['2F85 open']
        self.cloth = bpy.data.objects['gown.001']
        self.arm = bpy.data.objects['Armature']
        self.fps = bpy.context.scene.render.fps
        self.time = 70

        self.render_size = (int(1280), int(720))

        ###################################### change
        self.IMAGE_HEIGHT_OUT = 96
        self.SCALE = 720 / self.IMAGE_HEIGHT_OUT

    def __repr__(self):
        return f'GraspEnv(obs height={self.IMAGE_HEIGHT}, obs_width={self.IMAGE_WIDTH}, AS={self.action_space_type})'

    def get_observation(self):
        cnt = 0
        image_d_list = []
        pixel_list = []
        for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
            bpy.context.scene.camera = cam
            bpy.ops.render.render()
            print(cnt)

            ########################################################################################################
            # depsgraph = bpy.context.evaluated_depsgraph_get()
            # cloth_deformed = self.cloth.evaluated_get(depsgraph)
            # vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)]
            # camera_coord = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene,
            #                                                             bpy.context.scene.camera,
            #                                                             vertices[2800])
            # pixel = [round(camera_coord.x * self.render_size[0]),
            #          round(self.render_size[1] - camera_coord.y * self.render_size[1])]
            # pixel_list.append(pixel)
            ########################################################################################################

            bpy.data.images['Render Result'].save_render(
                filepath='/home/baxter-prl/Desktop/img/rgb%05d.png' % cnt)

            pixels = np.array(bpy.data.images['Viewer Node'].pixels)
            image = pixels.reshape(720, 1280, 4)
            image_d = image[:, :, 0:3]
            image_d_list.append(image_d)
            cnt += 1

        np.save('/home/baxter-prl/Desktop/img/d.npy', image_d_list)
        # np.save('/home/baxter-prl/Desktop/pixel.npy', pixel_list)
        # pixelsx = (pixel_list[:, 0] - 280) / SCALE
        # pixelsy = pixel_list[:, 1] / SCALE
        # pixelbox = []
        # for k in range(6):
        #     for i in range(int(pixelsx[k]), int(pixelsx[k]) + 15):
        #         for j in range(int(pixelsy[k]) - 10, int(pixelsy[k]) + 10):
        #             pixelbox.append([k, i, j])
        # pp = []
        # for k in range(len(pixelbox)):
        #     pp.append(pixelbox[k][1] * 96 + pixelbox[k][0])
        # ppfinal = []
        # for k in range(3):
        #     for i in range(len(pp)):
        #         ppfinal.append(int((pixelbox[i][0]*3+k) * 96 * 96 + pp[i]))
        # np.save('/home/baxter-prl/Desktop/ppfinal', ppfinal)

    def _env_reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        # if self.demo_mode:
        #     self.controller.stay(5000, render=self.render)

        # self.cloth.location = (0, 0, 10)
        self.c = self.arm.pose.bones["Bone"]
        self.c.location = (0, 0, 0)
        self.c.rotation_euler = (0, 0, 0)
        self.count = 0

    def _env_prepare_step(self, action):
        ###################################################################### remember to change
        if self.count <= 99:
            self.velocity = Vector((0, 0, 0))

        if self.count == 99:
            self.get_observation()

        if self.count == 100:
            x = action[0] % self.IMAGE_HEIGHT_OUT  # width
            y = action[0] // self.IMAGE_HEIGHT_OUT  # height
            self.rot_whichaction = action[1] % 3
            self.rot_whichcamera = action[1] // 3

            ###################################################################### change rot_whichcamera + 4
            self.rot_whichcameramove = self.rot_whichcamera + 4

            print(
                colored('pixel x {} pixel y {} rot_whichaction {} rot_whichcamera {}'.format(x, y, self.rot_whichaction,
                                                                                             self.rot_whichcamera),
                        color='yellow', attrs=['bold']))

            ######################################################################
            pixel_list = []
            cnn = 0
            for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
                if cnn == self.rot_whichcamera:
                    bpy.context.scene.camera = cam
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    cloth_deformed = self.cloth.evaluated_get(depsgraph)
                    vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)]
                    self.velocity_len_list = []
                    self.velocity_list = []
                    for i in range(len(vertices)):
                        camera_coord = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene,
                                                                                    bpy.context.scene.camera,
                                                                                    vertices[i])
                        pixel = [round(camera_coord.x * self.render_size[0]),
                                 round(self.render_size[1] - camera_coord.y * self.render_size[1])]

                        if np.absolute(y - pixel[1] / self.SCALE) < 5 and np.absolute(
                                x - (pixel[0] - 280) / self.SCALE) < 5:
                            self.pos = vertices[i]
                            dis = mathutils.Vector((
                                self.pos[0] - 0.5 * math.sin(self.rot_whichcameramove * 30 / 180 * math.pi) / 0.033,
                                self.pos[2] - 0, -(self.pos[1] - (
                                            -0.5 * math.cos(self.rot_whichcameramove * 30 / 180 * math.pi) / 0.033))))
                            self.velocity = dis / (self.time - 8)
                            self.velocity_len_list.append([self.velocity.length_squared])
                            self.velocity_list.append(self.velocity)

                    if len(self.velocity_list) != 0:
                        self.inx = np.argmin(self.velocity_len_list)
                cnn += 1

        if self.count >= 101:
            if len(self.velocity_list) != 0:
                self.velocity = self.velocity_list[self.inx]
            else:
                self.velocity = Vector((0, 0, 0))

        self._apply_motor_force()

    def _apply_motor_force(self, record_grasps=False, markers=False, action_info='no info'):
        """
        Lets the agent execute the action.

        Args:
            action: The action to be performed.

        Returns:
            observation: np-array containing the camera image data
            rewards: The reward obtained
            done: Flag indicating weather the episode has finished or not
            info: Extra info
        """

        done = False
        info = {}

        if 0 <= self.count <= 99:
            self.c.location += self.velocity

        if self.count == 100:
            self.c.location += Vector((
                0.5 * math.sin(self.rot_whichcameramove * 30 / 180 * math.pi) / 0.033,
                -10,
                -(-0.5 * math.cos(self.rot_whichcameramove * 30 / 180 * math.pi) / 0.033)))
            self.c.rotation_mode = 'XYZ'
            axis = 'Y'
            angle = self.rot_whichcameramove * 30
            self.c.rotation_euler.rotate_axis(axis, math.radians(angle))

        if 0 < self.count <= 162:
            self.c.location += self.velocity

        if 162 < self.count:
            if self.rot_whichaction == 0:
                self.c.location += Vector((self.velocity[2], 0, -self.velocity[0]))
            if self.rot_whichaction == 1:
                self.c.location += Vector((0, 0, 0))
            if self.rot_whichaction == 2:
                self.c.location += Vector((-self.velocity[2] * 1.9, 0, self.velocity[0] * 1.9))

        self.count = self.count + 1

    def _env_post_step(self):
        c = self.gripper.matrix_world.translation[0]  # just put here, mean nothing

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cloth_deformed = self.cloth.evaluated_get(depsgraph)
        vertices_clo = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)]
        sumdis_clo = 0
        verts_clo = [1371, 1371, 1371, 1374, 1375, 1376, 1377, 1378, 1484, 1485, 1486, 1487, 1360, 1362, 1365, 1364,
                     1363, 860, 861, 862, 863, 864, 829, 830, 831, 832, 833, 782, 784, 786, 788, 790, 781, 783, 785,
                     787, 789, 803, 805, 806, 807, 808, 809]
        for i in range(len(verts_clo)):
            l = np.sqrt(np.sum(np.square(vertices_clo[verts_clo[i]] - vertices_clo[2800])))
            sumdis_clo += l

        gripper_deformed = self.gripper.evaluated_get(depsgraph)
        vertices_gr = [gripper_deformed.matrix_world @ v.co for v in list(gripper_deformed.data.vertices)]
        sumdis_gr = np.sqrt(np.sum(np.square(vertices_gr[27729] - vertices_clo[2800])))

        if sumdis_clo > 193 and sumdis_gr < 1:
            reward = 1.0
        else:
            reward = 0.0

        if self.count == 169:
            print(colored('sumdis_clo {} sumdis_gr {} reward {}'.format(sumdis_clo, sumdis_gr, reward),
                          color='yellow', attrs=['bold']))

        return dict(
            obs=(c),
            reward=reward,
            done=bool(
                abs(c) > 100  # mean nothing
            )
        )

        # if self.demo_mode:
        #     self.controller.stay(200, render=render)
        #     return 'demo'

    # def close(self):
    #     mujoco_env.MujocoEnv.close(self)
    #     cv.destroyAllWindows()

    def print_info(self):
        print('Model timestep:', self.model.opt.timestep)
        print('Set number of frames skipped: ', self.frame_skip)
        print('dt = timestep * frame_skip: ', self.dt)
        print('Frames per second = 1/dt: ', self.metadata['video.frames_per_second'])
        print('Actionspace: ', self.action_space)
        print('Observation space:', self.observation_space)


def main():
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--render-every', default=None, type=int)
    parser.add_argument('--real-time', dest='realtime', action='store_true')
    parser.add_argument('--no-real-time', dest='realtime',
                        action='store_false')
    envargs = parser.parse_args(remainder)

    agent = btb.env.RemoteControlledAgent(
        args.btsockets['GYM'],
        real_time=envargs.realtime
    )
    env = PregraspEnv(agent)
    env.attach_default_renderer(every_nth=envargs.render_every)
    env.run(frame_range=(1, 10000), use_animation=True)


main()
