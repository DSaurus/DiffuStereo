from config.db_config import config_db as config
import argparse
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.dataset.DbDataset import TrainDatasetMVS
from lib.model.MVSModel import NeuMVSDModel, MVSModel
from lib.util.IO import customized_export_ply

def depth2pts(depth, extrinsic, intrinsic, normal=None):
    depth = depth[0, 0]
    S = depth.shape[0]
    valid = depth >= 0.2
    rot = extrinsic[0, 0, :3, :3]
    trans = extrinsic[0, 0, :3, 3:]
    intrinsic = intrinsic[0, 0]
    if normal is None:
        normal = depth[valid].unsqueeze(1).repeat(1, 3)
        normal[:, :2] = 0
        normal[:, 2] = 1
        normal = -(rot.T @ normal.T).T
    else:
        normal = normal[valid, :]
        normal = -(rot.T @ normal.T).T

    pts_2d = torch.zeros((S, S, 3)).to(depth.device)
    for i in range(S):
        pts_2d[i, :, 0] = torch.linspace(0, S, S)
        pts_2d[:, i, 1] = torch.linspace(0, S, S)
    pts_2d[:, :, 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, 0] -= intrinsic[0, 2]
    pts_2d[:, :, 1] -= intrinsic[1, 2]
    pts_2d[:, :, :2] = pts_2d[:, :, :2] * pts_2d[:, :, 2:]
    pts_2d = pts_2d[valid, :]

    pts_2d[:, 0] /= intrinsic[0, 0]
    pts_2d[:, 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T, normal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--use_db_normal', action='store_true')
    parser.add_argument('--view_list', nargs='+', type=int)
    parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, "cpu" for CPU mode')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.num_threads = 1
    cfg.gpus = arg.gpus
    cfg.batch_size = 1
    cfg.dataset.yaw_list = list(range(0, 360, 4))
    if arg.name is not None:
        cfg.name = arg.name
    if arg.dataroot is not None:
        cfg.dataset.dataroot = arg.dataroot
    cfg.freeze()

    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs('%s/%s' % (cfg.checkpoint_path, cfg.name), exist_ok=True)
    os.makedirs('%s/%s' % (cfg.result_path, cfg.name), exist_ok=True)

    dataset = TrainDatasetMVS(cfg.dataset, phase='train')
    
    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device('cuda:%d' % gpu_ids[0])
        db_net = NeuMVSDModel(cfg.mvs_model).to(device)
    else:
        gpu_ids = None
        db_net = NeuMVSDModel(cfg.mvs_model)
    mvs_model = MVSModel(cfg.mvs_model)

    if os.path.exists(cfg.load_db_net_checkpoint):
        print('db_net : loading from %s' % cfg.load_db_net_checkpoint)
        db_net.load_state_dict(torch.load(cfg.load_db_net_checkpoint), strict=False)
    
    db_net.eval()

    pts_list = []
    normal_list = []
    v_list = arg.view_list
    print(v_list)
    cam_nums = len(v_list)
    fid = 1
    use_db_normal = arg.use_db_normal
    for tid in range(0, cam_nums-1, 2):
        data = dataset.get_item(fid, view_ids=[v_list[tid], v_list[(tid+1)%cam_nums]])
        
        to_cuda = ['image', 'calib', 'mask', 'extrinsic',
            'intrinsic', 'depth', 'depth_db']
        for data_item in to_cuda:
            data[data_item] = data[data_item].unsqueeze(0).to(device=device)
        data = db_net.gen_data(data, train=False)
        with torch.no_grad():
            data = db_net.inference(data, infer=True)

        epi_map = data['epi_map']
        epi_mask = data['epi_mask']
        epi_mask = 1 - epi_mask.float().unsqueeze(0).unsqueeze(0)
        epi_mask = F.max_pool2d(epi_mask, 5, padding=2, stride=1)
        epi_mask = (1 - epi_mask)[0, 0] >= 0.5
        depth = data['pred_depth']
        pred_pos = data['pred_pos']
        depth[:, :, ~epi_mask] = 0
        t_map = data['t_map']

        img1 = data['img1']
        img1[:, :, ~epi_mask] = 0
        img1 = ((img1 * 0.5 + 0.5).clamp(0, 1)*255).detach().cpu()[0].permute(1, 2, 0).numpy().astype(int)
        img1 = img1[:, :, ::-1]

        q_img2 = data['q_img2']
        q_img1 = data['q_img1']
        q_img1 = ((q_img1 * 0.5 + 0.5).clamp(0, 1)*255).detach().cpu()[0].permute(1, 2, 0).numpy().astype(int)
        q_img2 = ((q_img2 * 0.5 + 0.5).clamp(0, 1)*255).detach().cpu()[0].permute(1, 2, 0).numpy().astype(int)
        cv2.imwrite('%s/%s/q1.png' % (cfg.result_path, cfg.name), q_img2[:, :, ::-1])
        cv2.imwrite('%s/%s/q2.png' % (cfg.result_path, cfg.name), q_img1[:, :, ::-1])
        
        if use_db_normal:
            normal = cv2.imread(
                os.path.join(cfg.dataset.dataroot, 'normal_db', '%s' % data['name'], '%d.png' % v_list[tid]))
            normal = torch.FloatTensor(((normal / 255)[:, :, ::-1]*2-1)).to(depth.device)
        else:
            normal = None
        pts, normal = depth2pts(depth, data['extrinsic'], data['intrinsic'], normal=normal)
        pts_list.append(pts)
        normal_list.append(normal)

    pts_list = torch.cat(pts_list, dim=0)
    normal_list = torch.cat(normal_list, dim=0)
    pts = pts_list.detach().cpu().numpy()
    normal = normal_list.detach().cpu().numpy()
    customized_export_ply('%s/%s/fusion%s.ply'  % (cfg.result_path, cfg.name, data['name']), pts, v_n=normal)

                
               
