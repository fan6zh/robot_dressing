'''
blender python
'''

import bpy, bpy_extras
import os
import sys
import json
import time
import math
import mathutils
import random
import bmesh
import numpy as np

from math import *
from mathutils import *
from random import sample


def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def set_renderer_properties(scene):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    # RGB resolution
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.compression = 100


def set_scene_properties(scene_name='Scene'):
    scene = bpy.data.scenes[scene_name]
    set_renderer_properties(scene)
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.system_rotation = 'RADIANS'
    render_size = (
        int(scene.render.resolution_x),
        int(scene.render.resolution_y),
    )
    return render_size


def add_camera():
    # bpy.ops.object.light_add(type='SUN', radius=100, location=(0, 0, 40))
    bpy.ops.object.camera_add(location=(17, 0, 27.37 + 1.21 / scale),
                              rotation=(math.pi / 180 * 20, 0, math.pi / 2))
    bpy.context.scene.camera = bpy.context.object
    bpy.context.object.data.lens = 1.8
    # RGB FOV
    bpy.context.object.data.sensor_width = 2.48
    bpy.context.object.data.sensor_height = 1.38


def build_nodes():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)

    rl_node = tree.nodes.new('CompositorNodeRLayers')
    view_node = tree.nodes.new(type="CompositorNodeViewer")
    inv_node = tree.nodes.new(type="CompositorNodeInvert")

    map_node = tree.nodes.new(type="CompositorNodeMapValue")
    map_node.size = [0.02]
    map_node.use_min = True
    map_node.use_max = True
    map_node.min = [0]
    map_node.max = [1]

    fileOutput_rgb = tree.nodes.new('CompositorNodeOutputFile')
    fileOutput_rgb.base_path = '/home/baxter-prl/Desktop/grasp/point6/rgb'
    fileOutput_rgb.file_slots[0].use_node_format = True
    fileOutput_rgb.format.color_mode = 'RGB'
    fileOutput_rgb.format.file_format = 'PNG'
    fileOutput_rgb.file_slots[0].path = 'rgb%05d_' % episode

    fileOutput_depth = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput_depth.base_path = '/home/baxter-prl/Desktop/grasp/point6/d'
    fileOutput_depth.file_slots[0].use_node_format = False
    fileOutput_depth.file_slots[0].format.file_format = "OPEN_EXR"
    fileOutput_depth.file_slots[0].path = 'd%05d_' % episode

    # # RGB
    # links.new(rl_node.outputs[0], fileOutput_rgb.inputs[0])

    # depth
    links.new(rl_node.outputs[2], map_node.inputs[0])
    links.new(map_node.outputs[0], inv_node.inputs[1])
    links.new(inv_node.outputs[0], view_node.inputs[0])
    links.new(rl_node.outputs[1], view_node.inputs[1])
    links.new(inv_node.outputs[0], fileOutput_depth.inputs[0])


def build_nodes_mask():
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)

    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_node = tree.nodes["Render Layers"]
    mask_node = tree.nodes.new(type="CompositorNodeIDMask")
    mask_node.index = 100

    fileOutput_mask = tree.nodes.new('CompositorNodeOutputFile')
    fileOutput_mask.base_path = '/home/baxter-prl/Desktop/grasp/point6/mask'
    fileOutput_mask.file_slots[0].use_node_format = False
    fileOutput_mask.file_slots[0].format.color_mode = "BW"
    fileOutput_mask.file_slots[0].path = 'rgb%05d_gown_0_' % episode

    links.new(render_node.outputs["IndexOB"], mask_node.inputs[0])
    links.new(mask_node.outputs[0], fileOutput_mask.inputs[0])

    # # Clean up
    # scene.render.engine = saved
    # for node in tree.nodes:
    #     if node.name != "Render Layers":
    #         tree.nodes.remove(node)
    # scene.use_nodes = False


def reset_manikin(manikin):
    manikin.location = (0, 0, 1)
    # manikin.rotation_euler = (1.57, 0, -1.57)


def reset_gripper(gripper):
    gripper.location = (3.8478, -5.4537, 32.603)
    gripper.rotation_euler = (-1.57, 0, 1.57)


def reset_bed(bed):
    bed.location = (0, 0, -2)


def reset_cloth(cloth):
    cloth.location = (0, 0, 0)
    # cloth.rotation_euler = (0, 0, 0)
    bpy.context.scene.frame_set(0)


def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode


def pattern(obj, texture_filename):
    '''Add image texture to object'''
    mat = bpy.data.materials.new(name="ImageTexture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_filename)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    set_viewport_shading('MATERIAL')
    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def save_img():
    for i in range(1, time):
        bpy.context.scene.frame_set(i)
        if i == 799:
            # depth / rgb / mask
            bpy.ops.render.render()

            # viewpoint rgb
            bpy.context.scene.render.image_settings.color_mode = "RGB"
            bpy.ops.render.opengl(write_still=True)
            bpy.data.images['Render Result'].save_render(
                filepath='/home/baxter-prl/Desktop/grasp/point6/rgb/rgb%05d.png' % episode)


if __name__ == '__main__':
    time = 800
    scale = 0.033
    episode_num = 200

    # for episode in range(episode_num):
    episode = 1
    clear_scene()
    add_camera()
    render_size = set_scene_properties()
    build_nodes_mask()

    # full_path_to_file = "/home/baxter-prl/z_cont/assets/2F85_closed.stl"
    # bpy.ops.import_mesh.stl(filepath=full_path_to_file)
    # gripper = bpy.context.selected_objects[0]
    # bpy.context.view_layer.objects.active = gripper
    # texture_filepath = '/home/baxter-prl/z_cont/texture/black.jpg'
    # pattern(gripper, texture_filepath)
    # gripper.scale = ((0.03, 0.03, 0.03))
    # reset_gripper(gripper)

    bpy.ops.wm.collada_import(filepath="/home/baxter-prl/z_cont/assets/manikin_l.dae")

    full_path_to_file = "/home/baxter-prl/z_cont/assets/bed.obj"
    bpy.ops.import_scene.obj(filepath=full_path_to_file)
    bed = bpy.context.selected_objects[0]
    bed.scale = ((34.5, 34.5, 34.5))
    reset_bed(bed)

    full_path_to_file = "/home/baxter-prl/z_cont/assets/gown_priorleftarm.obj"
    bpy.ops.import_scene.obj(filepath=full_path_to_file)
    cloth = bpy.context.selected_objects[0]
    cloth.pass_index = 100
    bpy.context.view_layer.objects.active = cloth
    texture_filepath = '/home/baxter-prl/z_cont/texture/bluereal.jpg'
    pattern(cloth, texture_filepath)
    reset_cloth(cloth)

    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         area.spaces[0].region_3d.view_perspective = 'CAMERA'
    #         for space in area.spaces:
    #             if space.type == 'VIEW_3D':
    #                 space.overlay.show_object_origins = False
    #                 space.overlay.show_bones = False

    bpy.ops.object.modifier_add(type='CLOTH')
    gripped_group = bpy.context.object.vertex_groups.new(name='Pinned')
    gripped_group.add([1754, 2246, 1819, 2313, 2299, 2301, 2270], 0.999, 'ADD')
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True
    bpy.context.object.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    bpy.context.object.modifiers["Cloth"].point_cache.frame_end = time
    bpy.context.object.modifiers["Cloth"].collision_settings.distance_min = 0.1
    bpy.context.object.modifiers["Cloth"].collision_settings.collision_quality = 5
    bpy.context.object.modifiers["Cloth"].settings.tension_stiffness = 1  # X_para[episode, 1]
    bpy.context.object.modifiers["Cloth"].settings.bending_stiffness = 0.05  # X_para[episode, 4]

    ###################################################################################################### armature cloth
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    arm = bpy.data.objects['Armature']
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = cloth

    for _ in range(2):
        bpy.ops.object.modifier_move_up(modifier="Armature")
    b_group = bpy.context.object.vertex_groups["Bone"]
    b_group.add([1754, 2246, 1819, 2313, 2299, 2301, 2270], 1.0, 'ADD')
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.posemode_toggle()
    frames = [0, 50, 300, 350, 550, 650, 750]
    # Two keyframes at the same spot for no application of armature movement.
    for b in arm.pose.bones:
        b.keyframe_insert("location", frame=frames[0])
        b.keyframe_insert("location", frame=frames[1])
    c = arm.pose.bones["Bone"]
    c.location += Vector((3.5, 8, 10))
    for b in arm.pose.bones:
        b.keyframe_insert("location", frame=frames[2])
        b.keyframe_insert("location", frame=frames[3])
    c.location += Vector((-2, 0, -20))
    for b in arm.pose.bones:
        b.keyframe_insert("location", frame=frames[4])
    c.location += Vector((-1.5, -4.5, -3))
    for b in arm.pose.bones:
        b.keyframe_insert("location", frame=frames[5])
    c.location += Vector((-2.5 + np.random.uniform(-0.5, 0.5, 1), 0, -13 + np.random.uniform(-1, 1, 1)))
    for b in arm.pose.bones:
        b.keyframe_insert("location", frame=frames[6])

    # ###################################################################################################### armature gripper
    # # Select the cloth _in_addition_ to the armature. The active one is the parent.
    # bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    # arm = bpy.data.objects['Armature.001']
    # gripper.select_set(True)
    # bpy.context.view_layer.objects.active = arm
    # bpy.ops.object.parent_set(type='ARMATURE_NAME')
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.context.view_layer.objects.active = gripper
    # bpy.ops.object.editmode_toggle()
    # bm = bmesh.from_edit_mesh(bpy.context.active_object.data)
    # verts = [v.index for v in bm.verts]
    # bpy.ops.object.mode_set(mode='OBJECT')
    #
    # for _ in range(2):
    #     bpy.ops.object.modifier_move_up(modifier="Armature")
    # b_group = bpy.context.object.vertex_groups["Bone"]
    # b_group.add(verts, 1.0, 'ADD')
    # bpy.context.view_layer.objects.active = arm
    # bpy.ops.object.posemode_toggle()
    # frames = [0, 50, 300, 350, 550, 650, 750]
    # # Two keyframes at the same spot for no application of armature movement.
    # for b in arm.pose.bones:
    #     b.keyframe_insert("location", frame=frames[0])
    #     b.keyframe_insert("location", frame=frames[1])
    # c = arm.pose.bones["Bone"]
    # c.location += Vector((3.5, 8, 10))
    # for b in arm.pose.bones:
    #     b.keyframe_insert("location", frame=frames[2])
    #     b.keyframe_insert("location", frame=frames[3])
    # c.location += Vector((-2, 0, -20))
    # for b in arm.pose.bones:
    #     b.keyframe_insert("location", frame=frames[4])
    # c.location += Vector((-1.5, -4.5, -3))
    # for b in arm.pose.bones:
    #     b.keyframe_insert("location", frame=frames[5])
    # c.location += Vector((-2.5 + np.random.uniform(-0.5, 0.5, 1), 0, -13 + np.random.uniform(-1, 1, 1)))
    # for b in arm.pose.bones:
    #     b.keyframe_insert("location", frame=frames[6])

    ###################################################################################################### armature manikin
    ob = bpy.data.objects['Armature.002']
    bpy.ops.object.posemode_toggle()
    pbone = ob.pose.bones['Bone.002']
    pbone.keyframe_insert(data_path="rotation_euler", frame=300)
    pbone.rotation_mode = 'XYZ'
    axis = 'X'
    angle = -30
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    # bpy.ops.object.mode_set(mode='OBJECT')
    # insert a keyframe
    pbone.keyframe_insert(data_path="rotation_euler", frame=350)
    pbone.keyframe_insert(data_path="rotation_euler", frame=750)
    pbone.rotation_mode = 'XYZ'
    axis = 'X'
    angle = 15
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    pbone.keyframe_insert(data_path="rotation_euler", frame=800)

    for obj in bpy.data.objects:
        obj.modifiers.new(type='COLLISION', name='collision')

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = time
    bpy.ops.screen.animation_play()
