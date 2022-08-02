from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image, ImageOps, ImageDraw
import cv2
import torch
from PIL.ImageFilter import GaussianBlur, MinFilter
import trimesh
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
from einops import rearrange
import torchvision
from lib.dataset.ImageModule import ImageModule

log = logging.getLogger('trimesh')
log.setLevel(40)

class TrainDatasetMVS(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase):
        self.opt = opt
        self.root = opt.dataroot
        self.RENDER = opt.img_path
        self.MASK = opt.mask_path
        self.DEPTH = opt.depth_path
        self.DEPTH_DB = opt.depth_db_path

        self.PARAM_EX = opt.param_ex_path
        self.PARAM_IN = opt.param_in_path
        self.OBJ = opt.obj_path

        self.image_module = ImageModule(opt)

        self.num_views = opt.num_views
        self.yaw_list = opt.yaw_list
        self.angle = opt.angle

        if phase == 'train':
            self.subjects = self.get_subjects(True)
        else:
            self.subjects = self.get_subjects(False)
        self.phase = phase

        self.jittor = torchvision.transforms.ColorJitter(brightness=opt.aug.bri, contrast=opt.aug.con, 
            saturation=opt.aug.sat, hue=opt.aug.hue)

    def get_subjects(self, is_train):
        all_subjects = os.listdir(os.path.join(self.root, 'img'))
        return sorted(list(set(all_subjects)))

    def __len__(self):
        if self.phase == 'test':
            return 1000
        return len(self.subjects) * len(self.yaw_list)

    def get_novel_depth(self, subject, view_ids, aug=True):
        calib_list = []
        extrinsic_list = []
        intrinsic_list = []
        depth_list = []
        depth_db_list = []
        render_list = []
        mask_list = []
        
        for vid in view_ids:
            extrinsic_path = os.path.join(self.root, self.PARAM_EX % (subject, vid))
            intrinsic_path = os.path.join(self.root, self.PARAM_IN % (subject, vid))
            depth_path = os.path.join(self.root, self.DEPTH % (subject, vid))
            depth_db_path = os.path.join(self.root, self.DEPTH_DB % (subject, vid))
            render_path = os.path.join(self.root, self.RENDER % (subject, vid))
            mask_path = os.path.join(self.root, self.MASK % (subject, vid))
           
            # loading calibration data
            extrinsic = np.load(extrinsic_path)
            intrinsic = np.load(intrinsic_path)
            depth = np.load(depth_path)['arr_0']
            depth_db = np.load(depth_db_path)
            if depth_db.__contains__('arr_0'):
                depth_db = depth_db['arr_0']
            else:
                depth_db = depth
            mask = Image.open(mask_path).convert('RGB')
            render = Image.open(render_path).convert('RGB')
            img_size = depth.shape[1]

            if self.opt.taichi_intrinsic:
                intrinsic[1, :] *= -1
                intrinsic[1, 2] += img_size
            extrinsic = torch.Tensor(extrinsic).float()
            intrinsic = torch.Tensor(intrinsic).float()
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

            mask = torch.FloatTensor((depth != 0))
            mask[mask > 1] = 1.0

            depth = torch.FloatTensor(depth)
            depth_db = torch.FloatTensor(depth_db)
            render = torch.FloatTensor(np.array(render)).permute(2, 0, 1) / 255
            if aug:
                render = self.jittor(render)
                render = render + torch.randn_like(render) * 0.02
            
            render = render * 2 - 1
            render = render * mask.reshape(1, img_size, img_size)

            render_list.append(render)
            mask_list.append(mask.reshape(1, img_size, img_size))
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            intrinsic_list.append(intrinsic)
            depth_list.append(depth.reshape(1, img_size, img_size))
            depth_db_list.append(depth_db.reshape((1, img_size, img_size)))

        return {
            'image': torch.stack(render_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'intrinsic': torch.stack(intrinsic_list, dim=0),
            'depth': torch.stack(depth_list, dim=0),
            'depth_db': torch.stack(depth_db_list, dim=0)
        }

    def get_item(self, index, view_ids=None):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)

        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject, subject + '.obj'),
            'sid': sid,
            'yid': yid,
        }
        aug = False
        
        if view_ids is None:
            vid1 = np.random.choice(self.yaw_list, 1)[0]
            vid2 = (vid1 + np.random.choice(range(self.angle-5, self.angle+5), 1)[0]) % 360
            view_ids = [vid1, vid2]
            aug = True
        train_data = self.get_novel_depth(subject, view_ids, aug=aug)
        res.update(train_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)