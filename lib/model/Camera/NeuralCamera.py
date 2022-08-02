import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm


class NeuralCamera():
    def __init__(self, db_model, opt):
        self.db_model = db_model
        self.samples = opt.samples
        self.ray_batch = opt.ray_batch
    
    @staticmethod
    def gen_full_rays(extrinsic, intrinsic, resolution):
        # resolution (width, height)
        rays_o_list = []
        rays_d_list = []
        rot = extrinsic[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, extrinsic[:, :3, 3:])
        c2w = torch.cat((rot, trans.reshape(-1, 3, 1)), dim=2)
        for b in range(intrinsic.shape[0]):
            fx, fy, cx, cy = intrinsic[b, 0, 0], intrinsic[b, 1, 1], intrinsic[b, 0, 2], intrinsic[b, 1, 2]
            W = resolution[b, 0].int().item()
            H = resolution[b, 1].int().item()
            i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[b, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[b, :3,-1].expand(rays_d.shape)
            rays_o_list.append(rays_o.unsqueeze(0))
            rays_d_list.append(rays_d.unsqueeze(0))
        rays_o_list = torch.cat(rays_o_list, dim=0)
        rays_d_list = torch.cat(rays_d_list, dim=0)
        # rays [B, C, H, W]
        return rearrange(rays_o_list, 'b h w c -> b c h w'), rearrange(rays_d_list, 'b h w c -> b c h w')
        
