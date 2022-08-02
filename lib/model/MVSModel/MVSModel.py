import torch
import torch.nn.functional as F
from einops import rearrange

import lib.model.Camera.NeuralCamera as NC
from lib.model.QueryModule import QueryModule

class MVSModel():
    def __init__(self, opt):
        self.query_module = QueryModule()
        self.D = 0.025
        self.noise = opt.noise
        self.scale = self.D / 2.5
    
    def tmap_to_z(self, rays_z, t_map, epi_mask):
        D = self.D
        k = (1 - t_map[:, :, epi_mask]).clamp(1e-8)
        z2 = rays_z + D
        z1 = (rays_z - D).clamp(1e-8)
        T = (z2 / k - z2) / z1
        t = T / (1 + T)
        t_z = t*z1 + (1-t)*z2
        return t_z
    
    def z_to_tmap(self, rays_z1, rays_z2, epi_mask):
        # represent rays_z1 using rays_z2 
        D = self.D
        t_map = torch.zeros_like(epi_mask).unsqueeze(0).unsqueeze(0).float()
        z1 = (rays_z2 - D).clamp(1e-8)
        z2 = rays_z2 + D
        t_z = rays_z1
        # t_z = t * z1 + (1-t)*z2
        t = (t_z - z2) / (z1 - z2)
        # t = T(1-t)
        T = t / (1-t).clamp(1e-8)
        # T = (z2 / k - z2) / z1
        k = z2 / (T*z1 + z2).clamp(1e-8)
        t_map[:, :, epi_mask] = 1 - k
        return t_map

    def tmap_to_depth(self, rays_z, t_map, epi_mask):
        D = self.D
        z_map = torch.zeros_like(t_map)
        k = (1 - t_map[:, :, epi_mask]).clamp(1e-8)
        z2 = rays_z + D
        z1 = (rays_z - D).clamp(1e-8)
        T = (z2 / k - z2) / z1
        t = T / (1 + T)
        t_z = t*z1 + (1-t)*z2
        z_map[:, :, epi_mask] = (1.0 / (t_z + 1e-8))
        return z_map
    
    def depth_to_world(self, depth, extrinsic, intrinsic, mask=True):
        B, C, H, W = depth.shape
        rays_o_grid, rays_d_grid = NC.gen_full_rays(extrinsic, intrinsic, torch.FloatTensor([[H, W]]).repeat(B, 1))
        rays_mask = depth[0, 0] != 0
        if mask:
            rays_z = 1.0 / (depth[:, :, rays_mask] + 1e-8)
            rays_o = rays_o_grid[:, :, rays_mask]
            rays_d = rays_d_grid[:, :, rays_mask]
            pts = rays_o + rays_d * rays_z
            return rays_mask, rays_o, rays_d, rays_z, pts
        else:
            rays_z = 1.0 / (depth[:, :, rays_mask] + 1e-8)
            rays_z_grid = torch.zeros_like(rays_o_grid[:, :1])
            rays_z_grid[:, :, rays_mask] = rays_z
            return rays_o_grid, rays_d_grid, rays_z_grid
        
    def project_valid(self, pts, depth2, calib2):
        B, C, H, W = depth2.shape
        pts_z, pts_2d = self.query_module.query_image(depth2, calib2, pts, (H, W), rt_pts_2d=True) # H W or W H ?
        pts_z = 1.0 / (pts_z + 1e-8)
        valid = torch.abs(pts_2d[0, 2, :] - pts_z[0, 0, :]) < 0.005
        return pts_2d, valid

    def get_epi_map(self, depth, rays_o, rays_d, rays_z, rays_mask, calib2, rt_pts2d=False, H=None):
        if H is None:
            B, C, H, W = depth.shape
        else:
            H, W = H, H
        D = self.D
        rays_z0 = rays_z - D
        pts0 = rays_o + rays_d * rays_z0
        rays_z1 = rays_z + D
        pts1 = rays_o + rays_d * rays_z1
        pts0_2d = self.query_module.camera.perspective(pts0, calib2)[:, :2, :]
        pts1_2d = self.query_module.camera.perspective(pts1, calib2)[:, :2, :]
        if rt_pts2d:
            pts2d = rays_o + rays_d * rays_z
            pts2d = self.query_module.camera.perspective(pts2d, calib2)[:, :2, :]
        pts0[:, 0, :] = (pts0_2d[:, 0, :] / W * 2) - 1
        pts0[:, 1, :] = (pts0_2d[:, 1, :] / H * 2) - 1
        pts1[:, 0, :] = (pts1_2d[:, 0, :] / W * 2) - 1
        pts1[:, 1, :] = (pts1_2d[:, 1, :] / H * 2) - 1
        epi_map = torch.zeros_like(depth).repeat(1, 4, 1, 1)
        epi_map[:, :2, rays_mask] = pts0[:, :2]
        epi_map[:, 2:4, rays_mask] = pts1[:, :2]
        t_map = torch.zeros_like(depth)
        t_map[:, :, rays_mask] = rays_z0 / (rays_z0 + rays_z1).clamp(1e-6)
        if rt_pts2d:
            return epi_map, t_map, pts2d
        return epi_map, t_map
    
    
    def query_image_pos(self, img, pos_map, epi_mask):
        # B = 1!!!
        B, C, H, W = img.shape
        pts = pos_map[:, :, epi_mask]
        uv = rearrange(pts, 'b (t c) n -> b n t c', t=1)
        # print(uv)
        samples = torch.nn.functional.grid_sample(img, uv, align_corners=True, mode='bilinear')
        pts_feature = rearrange(samples, 'b c n t -> b c (n t)')
        q_img = torch.zeros_like(pos_map[:, :1]).repeat(1, C, 1, 1)
        q_img[:, :, epi_mask] = pts_feature
        return q_img, None
    
    def posmap_to_tmap(self, pos, epi_map, epi_mask):
        tmap = torch.zeros_like(pos[:, :1])
        pos_pts = pos[:, :, epi_mask]
        epi_pts = epi_map[:, :, epi_mask]
        tmap[:, :, epi_mask] = (pos_pts[:, :1] - epi_pts[:, 2:3]) / (epi_pts[:, :1] - epi_pts[:, 2:3])
        return tmap.clamp(0, 1)

    def query_image(self, img, epi_map, t_map, epi_mask):
        # B = 1!!!
        B, C, H, W = img.shape
        epi_pts = epi_map[:, :, epi_mask]
        t_map_mask = t_map[:, :, epi_mask]
        pts = t_map_mask * epi_pts[:, :2] + (1 - t_map_mask) * epi_pts[:, 2:]
        uv = rearrange(pts, 'b (t c) n -> b n t c', t=1)
        # print(uv)
        samples = torch.nn.functional.grid_sample(img, uv, align_corners=True, mode='bilinear')
        pts_feature = rearrange(samples, 'b c n t -> b c (n t)')
        q_img = torch.zeros_like(t_map).repeat(1, C, 1, 1)
        q_img[:, :, epi_mask] = pts_feature
        pos_map = torch.zeros_like(t_map).repeat(1, 2, 1, 1)
        pos_map[:, :, epi_mask] = pts
        return q_img, pos_map
        
    def build_mvs(self, img1, depth1, ex1, in1, img2, depth2, ex2, in2):
        # project img1 to img2 and get mvs representation
        B, C, H, W = img1.shape
        # B = 1!!!
        rays_mask, rays_o, rays_d, rays_z, pts = self.depth_to_world(depth1, ex1, in1)
        # query_image(self, feature, calibs, pts, img_size)

        calib2 = torch.matmul(in2, ex2)
        pts_2d, valid = self.project_valid(pts, depth2, calib2)
        rays_mask[rays_mask.clone()] = valid
        rays_o = rays_o[:, :, valid]
        rays_d = rays_d[:, :, valid]
        rays_z = rays_z[:, :, valid]
        # epipolar
        epi_map, t_map = self.get_epi_map(depth1, rays_o, rays_d, rays_z, rays_mask, calib2)
        gt_pos = torch.zeros_like(epi_map[:, :2])
        gt_pos[:, :, rays_mask] = pts_2d[:, :2, valid]

        return rays_mask, epi_map, t_map, gt_pos