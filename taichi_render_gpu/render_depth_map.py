import taichi as ti
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import os
import sys
import cv2
import trimesh
import matplotlib.pyplot as plt
import math


def render_depth(dataroot, obj_path, obj_format, res=(2048, 2048), angles=range(360), flip_y=True, cover=False, is_4k=False, is_2k=False):
    depth_save_path = os.path.join(dataroot, 'depth_db')
    os.makedirs(depth_save_path, exist_ok=True)
    parameter_path = os.path.join(dataroot, 'parameter')

    for obj_name in tqdm(sorted(os.listdir(parameter_path))):
        print(obj_name)
        ti.init(ti.cpu)
        scene = t3.Scene()
        light = t3.Light()
        scene.add_light(light)
        depth_path = os.path.join(dataroot, 'depth_db', obj_name)
        normal_save_path = os.path.join(dataroot, 'normal_db', obj_name)
        if not cover:
            if os.path.exists(depth_path) and len(os.listdir(os.path.join(depth_path))) == len(angles):
                continue
        if not os.path.exists(os.path.join(obj_path, obj_format % obj_name)):
            print('Obj not found!')
            continue
        obj = t3.readobj(os.path.join(obj_path, obj_format % obj_name))
        model = t3.Model(obj=obj)
        scene.add_model(model)
        if is_4k:
            res = (4096, 4096)
        if is_2k:
            res = (2048, 2048)
        camera = t3.Camera(res=res)
        scene.add_camera(camera)

        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(normal_save_path, exist_ok=True)
        for angle in angles:
            intrinsic = np.load(os.path.join(parameter_path, obj_name, '{}_intrinsic.npy'.format(angle)))
            extrinsic = np.load(os.path.join(parameter_path, obj_name, '{}_extrinsic.npy'.format(angle)))
            if flip_y:
                camera.set_intrinsic(fx=intrinsic[0, 0], fy=-intrinsic[1, 1], cx=intrinsic[0, 2],
                                     cy=res[0] - intrinsic[1, 2])
            else:
                camera.set_intrinsic(fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2])

            trans = extrinsic[:, :3]
            T = extrinsic[:, 3]
            p = -trans.T @ T
            camera.set_extrinsic(trans.T, p)
            camera._init()
            scene.render()

            depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)[::-1, :]
            print(np.min(depth_map), np.max(depth_map))
            np.savez(os.path.join(depth_path, '{}.npz'.format(angle)), depth_map)
            ti.imwrite(camera.normal_map, os.path.join(normal_save_path, '{}.png'.format(angle)))



if __name__ == '__main__':
    res = (512, 512)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--obj_path", type=str)
    parser.add_argument("--obj_format", type=str)
    parser.add_argument("--yaw_list", type=int, nargs='+', default=[i for i in range(360)])
    parser.add_argument("--cover", action="store_true")
    parser.add_argument("--is_4k", action="store_true")
    parser.add_argument("--is_2k", action="store_true")
    args = parser.parse_args()

    render_depth(args.dataroot, args.obj_path, args.obj_format, angles=args.yaw_list, is_2k=args.is_2k, is_4k=args.is_4k)
