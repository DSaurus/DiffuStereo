import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from kornia.filters import sobel
from lib.network.ImageEncoder.unet_att import UNet as UnetATT
from lib.model.MVSModel import MVSModel
from tqdm import tqdm
import math

class NeuMVSDModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.feature_net_c = UnetATT(11, 32, channel_mults=(1, 2, 4, 8, 8),
                                   inner_channel=32, norm_groups=16, res_blocks=3)
        self.feature_conv = nn.Conv2d(32, 1, 1, padding=0)
        self.mask_net1 = UnetATT(11+32, 1, channel_mults=(1, 2, 4, 8, 8),
                                inner_channel=32, norm_groups=8, res_blocks=3)
        self.flow_net3 = UnetATT(11+32, 1, channel_mults=(1, 2, 4, 8, 8),
                                 inner_channel=32, norm_groups=16, res_blocks=3, with_noise_level_emb=opt.time)
        
        self.mvs_model = MVSModel(opt)

        if opt.is_2k:
            self.RS = 2048
        else:
            self.RS = 4096
        self.steps = opt.steps

    def reverse_diffusion(self, data):
        depth = data['depth']
        rays_o, rays_d, rays_z = data['rays_o'], data['rays_d'], data['rays_z']
        epi_map, epi_mask = data['epi_map'], data['epi_mask']
        img1, img2 = data['img1'], data['img2']
        c_feature = data['c_feature']

        scale = self.mvs_model.scale
        rays_z_o = rays_z.clone()
        noise = (0.25*torch.randn_like(rays_z))*self.mvs_model.scale
        rays_z = rays_z_o + noise
        input_delta_z = rays_z - rays_z_o
        epi_map, t_map = self.mvs_model.get_epi_map(depth, rays_o, rays_d, rays_z, epi_mask,
                                                    data['calib2'], H=self.RS)
        q_img2, pos = self.mvs_model.query_image(
            img2, epi_map.detach(), t_map, epi_mask)
        
        with torch.no_grad():
            for t in tqdm(range(self.steps, 0, -1)):
                z = (0.25*torch.randn_like(rays_z))*self.mvs_model.scale
                c0 = t / self.steps
                c1 = (t-1) / self.steps

                t_map = self.predict(epi_map, epi_mask, t_map, img1, q_img2, c_feature, input_delta_z, torch.FloatTensor([c0]).to(t_map.device))
                q_img2, pos_t = self.mvs_model.query_image(img2, epi_map.detach(), t_map, epi_mask)
                rays_z = self.mvs_model.tmap_to_z(rays_z, t_map, epi_mask)
                
                new_delta_z = rays_z - rays_z_o
                dc = c0 - c1

                # skip the last two steps
                if t < 3:
                    input_delta_z = (dc * new_delta_z + c1 * input_delta_z) / c0
                    rays_z = rays_z_o + input_delta_z
                    break
                else:
                    input_delta_z = (dc * new_delta_z + c1 * input_delta_z) / c0 + math.sqrt(dc * c1 / c0) * z
                    rays_z = rays_z_o + input_delta_z

                epi_map, t_map = self.mvs_model.get_epi_map(depth, rays_o, rays_d, rays_z, epi_mask,
                                                            data['calib2'], H=self.RS)
                q_img2, pos = self.mvs_model.query_image(img2, epi_map.detach(), t_map, epi_mask)
                
        epi_map, t_map = self.mvs_model.get_epi_map(depth, rays_o, rays_d, rays_z, epi_mask,
                                                    data['calib2'], H=self.RS)
        q_img2, pos = self.mvs_model.query_image(img2, epi_map.detach(), t_map, epi_mask)
        return q_img2, pos, epi_map, t_map, rays_z
    
    def epi_to_flow(self, epi_map, epi_mask):
        B, C, H, W = epi_map.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=epi_map.device), torch.linspace(-1, 1, W, device=epi_map.device))
        ori_grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        flow_map = epi_map - ori_grid
        flow_map[:, :, ~epi_mask] = 0
        flow_map = (flow_map).clamp(-1, 1)
        return flow_map

    def predict(self, epi_map, epi_mask, t_map, img1, q_img2, c_feature, input_delta_z, ct):
        flow_map = self.epi_to_flow(epi_map[:, :2], epi_mask)
        sobel_map = (
            sobel(flow_map, normalized=False)*20).clamp(0, 1)
        input_feature = torch.cat([flow_map.detach(), sobel_map.detach(
        ), t_map.detach(), img1, q_img2.detach(), c_feature], dim=1)
        input_feature[torch.isnan(input_feature)] = 0
        input_feature[torch.isinf(input_feature)] = 0
        t_map = nn.functional.sigmoid(self.flow_net3(input_feature, ct))
        return t_map
    
    def predict_global(self, epi_map, epi_mask, t_map, img1, q_img2):
        flow_map = self.epi_to_flow(epi_map[:, :2], epi_mask)
        sobel_map = (sobel(flow_map, normalized=False)*20).clamp(0, 1)
        input_feature = torch.cat(
            [flow_map, sobel_map, t_map, img1, q_img2], dim=1)
        input_feature[torch.isnan(input_feature)] = 0
        input_feature[torch.isinf(input_feature)] = 0
        input_feature = F.interpolate(
            input_feature, size=(self.RS // 4, self.RS // 4), mode='bilinear')
        c_feature = self.feature_net_c(input_feature)
        res = F.sigmoid(self.feature_conv(c_feature))
        res = F.interpolate(res, size=(self.RS, self.RS), mode='bilinear')
        return c_feature, res

    def predict_total(self, data):
        img1 = data['img1']
        img2 = data['img2']
        epi_map = data['epi_map']
        epi_mask = data['epi_mask']
        t_map = data['t_map']
        crop_x, crop_y = data['crop_x'], data['crop_y']
        rays_o, rays_d, rays_z, rays_z_gt = data['rays_o'], data['rays_d'], data['rays_z'], data['rays_z_gt']
        q_img2, pos0 = self.mvs_model.query_image(
            img2, epi_map, t_map, epi_mask)
        
        _, t_map_c = self.predict_global(epi_map, epi_mask, t_map, img1, q_img2)
        q_img2, pos_c = self.mvs_model.query_image(
            img2, epi_map.detach(), t_map_c, epi_mask)
        rays_z = self.mvs_model.tmap_to_z(rays_z, t_map_c, epi_mask)
        
        epi_map, t_map = self.mvs_model.get_epi_map(data['depth1'], rays_o, rays_d, rays_z, epi_mask,
                                                            data['calib2'], H=self.RS)
        c_feature, _ = self.predict_global(epi_map, epi_mask, t_map, img1, q_img2)
        
        data.update({
            'pos_mapc': pos_c,
            'c_feature': c_feature,
            'rays_z_c': rays_z
        })
        
        # add noise
        scale = self.mvs_model.scale
        noise = (0.25*torch.randn_like(rays_z))*scale
        delta_z = (rays_z_gt - rays_z)
        diff_mask = torch.abs(delta_z) < 0.025
        diff_gt_mask = data['epi_mask'].clone()
        diff_gt_mask[diff_gt_mask] = diff_mask
        t = self.var_sched.uniform_sample_t(1)
        c0 = self.var_sched.alpha_bars[t]
        input_delta_z = c0 * delta_z + torch.sqrt(1-c0) * noise
        rays_z = rays_z + input_delta_z 
        
        epi_map, t_map = self.mvs_model.get_epi_map(data['depth1'], rays_o, rays_d, rays_z, epi_mask,
                                                            data['calib2'], H=self.RS)
        q_img2, _ = self.mvs_model.query_image(img2, epi_map.detach(), t_map.detach(), epi_mask)

        C_RS = self.RS // 4
        pred_depth = torch.zeros_like(data['depth1'])
        pred_depth_up = torch.zeros_like(data['gt_pos'])
        mask_map_up = torch.zeros_like(pred_depth_up)
        pos_map_up = pred_depth_up.clone()
        weight_map = torch.zeros_like(pred_depth_up)
        x1, y1 = torch.meshgrid(torch.linspace(-1, 1, C_RS, device=pos_c.device),torch.linspace(-1, 1, C_RS, device=pos_c.device))
        weight_map_crop = torch.cat((x1.unsqueeze(0), y1.unsqueeze(0)), dim=0).unsqueeze(0)
        weight_map_crop = 2.01 - (torch.sum(weight_map_crop**2, dim=1).unsqueeze(1))
        # print(weight_map_crop)

        c_feature = F.interpolate(c_feature, size=(self.RS, self.RS))

        epi_map_o, t_map_o, img1_o, q_img2_o, c_feature_o, pred_depth_o = epi_map, t_map, img1, q_img2, c_feature, pred_depth
        rays_o_o, rays_d_o, rays_z_o, in_z_o = rays_o, rays_d, rays_z, input_delta_z
        epi_mask_o = epi_mask

        crop_list = [epi_map_o, t_map_o, img1_o, q_img2_o, c_feature_o, pred_depth_o]
        for i in range(len(crop_list)):
            crop_list[i] = crop_list[i][:, :,
                                    crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
        epi_map, t_map, img1, q_img2, c_feature, pred_depth = crop_list
        
        epi_crop_mask = torch.zeros_like(epi_mask_o)
        epi_crop_mask[crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS] = epi_mask_o[crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
        epi_crop_mask = epi_crop_mask[epi_mask_o]
        
        crop_list = [rays_o_o, rays_d_o, rays_z_o, in_z_o]
        for i in range(len(crop_list)):
            crop_list[i] = crop_list[i][:, :, epi_crop_mask]
        rays_o, rays_d, rays_z, input_delta_z = crop_list
        epi_mask = epi_mask_o[crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]

        
        t_map = self.predict(epi_map, epi_mask, t_map, img1, q_img2, c_feature, input_delta_z, c0.reshape((1)))
        q_img2, pos = self.mvs_model.query_image(
            img2, epi_map.detach(), t_map, epi_mask)
        rays_z = self.mvs_model.tmap_to_z(rays_z, t_map, epi_mask)
        epi_map, t_map = self.mvs_model.get_epi_map(pred_depth, rays_o, rays_d, rays_z, epi_mask,
                                                    data['calib2'], H=self.RS)
        
        mask_map = torch.zeros_like(pred_depth_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x+C_RS])
        pos_map_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS][:, :, epi_mask] += pos[:, :, epi_mask] * weight_map_crop[:, :, epi_mask]
        pred_depth_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS][:, :, epi_mask] += 1.0 / rays_z.clamp(1e-8) * weight_map_crop[:, :, epi_mask]
        weight_map[:, :, crop_y:crop_y+C_RS, crop_x:crop_x+C_RS][:, :, epi_mask] += weight_map_crop[:, :, epi_mask]
        pred_depth_up /= weight_map.clamp(1e-8)
        pos_map_up /= weight_map.clamp(1e-8)
        mask_map_up /= weight_map.clamp(1e-8)

        data.update({
            't_map': t_map,
            'pos_map': pos,
            'mask_map': mask_map,
            'diff_gt_mask': diff_gt_mask,
            'rays_z': rays_z,
            'mask_map_up': mask_map_up,
            'pred_depth': pred_depth_up,
            'pred_pos': pos_map_up,
            'q_img2': q_img2,
            'q_img1': img1,
            'sample_t': t
        })
        return data

    def gen_data(self, data, train=True):
        imgs = data['image']
        img1 = imgs[:, 0]
        img2 = imgs[:, 1]
        depth = data['depth_db']
        depth1 = depth[:, 0]
        depth2 = depth[:, 1]
        depth_gt = data['depth']
        depth1_gt = depth_gt[:, 0]
        depth2_gt = depth_gt[:, 1]
        extrinsic = data['extrinsic']
        ex1 = extrinsic[:, 0]
        ex2 = extrinsic[:, 1]
        intrinsic = data['intrinsic']
        in1 = intrinsic[:, 0]
        in2 = intrinsic[:, 1]
        epi_mask1, epi_map, t_map, gt_pos = self.mvs_model.build_mvs(img1, depth1_gt, ex1, in1,
            img2, depth2_gt, ex2, in2)
        epi_mask, epi_map, t_map, _ = self.mvs_model.build_mvs(img1, depth1, ex1, in1,
            img2, depth2, ex2, in2)
        epi_mask[~epi_mask1] = 0
        depth_mask = torch.abs(1.0/depth1_gt.clamp(1e-6) - 1.0/depth1.clamp(1e-6))[0, 0] < 0.025
        epi_gt_mask = epi_mask.clone()
        epi_gt_mask[~depth_mask] = 0
        gt_pos[:, :, ~depth_mask] = 0
        if train:
            epi_mask[~epi_gt_mask] = 0

        index_list = (epi_gt_mask.nonzero(as_tuple=True))
        t = torch.randint(index_list[0].shape[0], (1, 1)).item()
        C_RS = self.RS // 4
        if not train:
            min_crop_y = torch.min(index_list[0]).item()
            max_crop_y = torch.max(index_list[0]).item()
            min_crop_x = torch.min(index_list[1]).item()
            max_crop_x = torch.max(index_list[1]).item()
            process_list = [min_crop_x, min_crop_y, max_crop_x, max_crop_y]
            for i in range(2, 4):
                process_list[i] = min(self.RS-C_RS-1, max(process_list[i]-C_RS, 0))
            min_crop_x, min_crop_y, max_crop_x, max_crop_y = process_list
            crop_y_list = []
            crop_x_list = []
            SZ = C_RS // 2
            for c_y in range(min_crop_y, max_crop_y, SZ):
                for c_x in range(min_crop_x, max_crop_x, SZ):
                    crop_y_list.append(c_y)
                    crop_x_list.append(c_x)
                crop_y_list.append(c_y)
                crop_x_list.append(max_crop_x)
            for c_x in range(min_crop_x, max_crop_x, SZ):
                crop_y_list.append(max_crop_y)
                crop_x_list.append(c_x)
            crop_y_list.append(max_crop_y)
            crop_x_list.append(max_crop_x)
            # print(crop_x_list)
            # exit(0)
            data.update({'crop_y_list': torch.LongTensor(crop_y_list), 'crop_x_list': torch.LongTensor(crop_x_list)})
        else:
            crop_y, crop_x = index_list[0][t].item(), index_list[1][t].item()
            crop_y = min(self.RS-C_RS-1, max(crop_y-C_RS//2, 0))
            crop_x = min(self.RS-C_RS-1, max(crop_x-C_RS//2, 0))
            data.update({'crop_y': crop_y, 'crop_x': crop_x})

        B, C, H, W = img1.shape
        rays_o_grid, rays_d_grid, rays_z_grid = self.mvs_model.depth_to_world(depth1_gt, ex1, in1, mask=False)
        _, _, rays_z_gt = rays_o_grid[:, :, epi_mask], rays_d_grid[:, :, epi_mask], rays_z_grid[:, :, epi_mask]
        
        depth_d = depth1
        rays_o_grid, rays_d_grid, rays_z_grid = self.mvs_model.depth_to_world(depth_d, ex1, in1, mask=False)
        rays_o, rays_d, rays_z = rays_o_grid[:, :, epi_mask], rays_d_grid[:, :, epi_mask], rays_z_grid[:, :, epi_mask]
        calib2 = torch.matmul(in2, ex2)
        
        epi_map, t_map = self.mvs_model.get_epi_map(depth_d, rays_o, rays_d, rays_z, epi_mask, calib2)
        
        data.update({
            'img1': img1, 'img2': img2,
            'epi_map': epi_map, 
            'epi_mask': epi_mask, 'epi_gt_mask': epi_gt_mask,
            't_map': t_map,
            'gt_pos': gt_pos,
            'rays_o': rays_o,
            'rays_z': rays_z,
            'rays_z_gt': rays_z_gt,
            'rays_d': rays_d,
            'calib2': calib2,
            'depth1': depth1,
            'depth1_gt': depth1_gt,
            'sample_t': t
        })
        return data

    def inference(self, data, infer=False):
        img1 = data['img1']
        img2 = data['img2']
        epi_map = data['epi_map']
        epi_mask = data['epi_mask']
        t_map = data['t_map']
        if not infer:
            crop_x, crop_y = data['crop_x'], data['crop_y']
        rays_o, rays_d, rays_z = data['rays_o'], data['rays_d'], data['rays_z']
        q_img2, pos0 = self.mvs_model.query_image(
            img2, epi_map, t_map, epi_mask)
        data.update({
            'pos_map0': pos0
        })
        c_feature, t_map_c = self.predict_global(epi_map, epi_mask, t_map, img1, q_img2)
        _, pos_c = self.mvs_model.query_image(
            img2, epi_map.detach(), t_map_c, epi_mask)
        rays_z = self.mvs_model.tmap_to_z(rays_z, t_map_c, epi_mask)
        epi_map, t_map = self.mvs_model.get_epi_map(data['depth1'], rays_o, rays_d, rays_z, epi_mask,
                                                            data['calib2'], H=self.RS)
        q_img2, _ = self.mvs_model.query_image(img2, epi_map.detach(), t_map.detach(), epi_mask)
        c_feature, _ = self.predict_global(epi_map, epi_mask, t_map, img1, q_img2)

        C_RS = self.RS // 4
        pred_depth = torch.zeros_like(data['depth1'])
        pred_depth_up = torch.zeros_like(data['gt_pos'])
        mask_map_up = torch.zeros_like(pred_depth_up)
        pos_map_up = pred_depth_up.clone()
        weight_map = torch.zeros_like(pred_depth_up)
        x1, y1 = torch.meshgrid(torch.linspace(-1, 1, C_RS, device=pos_c.device),torch.linspace(-1, 1, C_RS, device=pos_c.device))
        weight_map_crop = torch.cat((x1.unsqueeze(0), y1.unsqueeze(0)), dim=0).unsqueeze(0)
        weight_map_crop = (0.95 - (torch.max(torch.abs(weight_map_crop), dim=1)[0].unsqueeze(1))).clamp(1e-6)
        # print(weight_map_crop)

        c_feature = F.interpolate(c_feature, size=(self.RS, self.RS))
        if infer:
            crop_x_list = data['crop_x_list']
            crop_y_list = data['crop_y_list']
            # print(crop_x_list)
            crop_list_len = crop_x_list.shape[0]
        else:
            crop_x_list = [crop_x]
            crop_y_list = [crop_y]
            crop_list_len = 1

        epi_map_o, t_map_o, img1_o, q_img2_o, c_feature_o, pred_depth_o = epi_map, t_map, img1, q_img2, c_feature, pred_depth
        rays_o_o, rays_d_o, rays_z_o = rays_o, rays_d, rays_z
        epi_mask_o = epi_mask

        pos_list = []
        for ind in range(crop_list_len):
            crop_x = crop_x_list[ind]
            crop_y = crop_y_list[ind]
            crop_list = [epi_map_o, t_map_o, img1_o, q_img2_o, c_feature_o, pred_depth_o]
            for i in range(len(crop_list)):
                crop_list[i] = crop_list[i][:, :,
                                        crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
            epi_map, t_map, img1, q_img2, c_feature, pred_depth = crop_list
        
            epi_crop_mask = torch.zeros_like(epi_mask_o)
            epi_crop_mask[crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS] = epi_mask_o[crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
            epi_crop_mask = epi_crop_mask[epi_mask_o]
            
            crop_list = [rays_o_o, rays_d_o, rays_z_o]
            for i in range(len(crop_list)):
                crop_list[i] = crop_list[i][:, :, epi_crop_mask]
            rays_o, rays_d, rays_z = crop_list
            epi_mask = epi_mask_o[crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
            if torch.sum(epi_mask) == 0:
                continue

            gt = data['gt_pos']
            gt[torch.isnan(gt)] = 0
            gt[torch.isinf(gt)] = 0
            gt_mask = data['epi_gt_mask']
            gt = gt[:, :, crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]
            gt_mask = gt_mask[crop_y:crop_y+C_RS, crop_x:crop_x+C_RS]

            reverse_data = {
                'depth': pred_depth, 
                'rays_o': rays_o, 'rays_d': rays_d, 'rays_z': rays_z,
                'epi_map': epi_map, 'epi_mask': epi_mask,
                'img1': img1, 'img2': img2, 'c_feature': c_feature, 'calib2': data['calib2'],
                'gt_mask': gt_mask, 'gt': gt
            }
            
            q_img2, pos, epi_map, t_map, rays_z = self.reverse_diffusion(reverse_data)
            
            mask_map = torch.zeros_like(pred_depth_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x+C_RS])
            pos_map_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS][:, :, epi_mask] += pos[:, :, epi_mask] * weight_map_crop[:, :, epi_mask]
            pred_depth_up[:, :, crop_y:crop_y+C_RS, crop_x:crop_x +
                        C_RS][:, :, epi_mask] += 1.0 / rays_z.clamp(1e-8) * weight_map_crop[:, :, epi_mask]
            weight_map[:, :, crop_y:crop_y+C_RS, crop_x:crop_x+C_RS][:, :, epi_mask] += weight_map_crop[:, :, epi_mask]
        pred_depth_up /= weight_map.clamp(1e-8)
        pos_map_up /= weight_map.clamp(1e-8)
        mask_map_up /= weight_map.clamp(1e-8)

        data.update({
            't_map': t_map,
            'pos_map': pos,
            'mask_map': mask_map,
            'rays_z': rays_z,
            'mask_map_up': mask_map_up,
            'pred_depth': pred_depth_up,
            'pred_pos': pos_map_up,
            'q_img2': q_img2,
            'q_img1': img1
        })
        return data
