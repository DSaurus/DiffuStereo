import torch
from einops import rearrange

from lib.model.Camera.Camera import Camera

class QueryModule():
    def __init__(self) -> None:
        self.camera = Camera()

    def query_mv_image(self, mv_feature, calibs, pts, img_size, rt_pts_2d=False):
        B, V, C, H, W = mv_feature.shape
        mv_feature_list = []
        pts_2d_list = []
        for v in range(V):
            if rt_pts_2d:
                pts_feature, pts_2d = self.query_image(mv_feature[:, v], calibs[:, v], 
                    pts, img_size, rt_pts_2d=True)
                pts_2d_list.append(pts_2d)
            else:
                pts_feature = self.query_image(mv_feature[:, v], calibs[:, v], pts, img_size)
            mv_feature_list.append(pts_feature)
        mv_feature_list = torch.stack(mv_feature_list, dim=1)
        if rt_pts_2d:
            pts_2d_list = torch.stack(pts_2d_list, dim=1)
            return mv_feature_list, pts_2d_list
        return mv_feature_list

    def query_image(self, feature, calibs, pts, img_size, rt_pts_2d=False):
        # img size (W, H)
        pts_2d = self.camera.perspective(pts, calibs) # [B, 3, N]
        pts_2d[:, 0, :] = (pts_2d[:, 0, :] / img_size[0] * 2) - 1
        pts_2d[:, 1, :] = (pts_2d[:, 1, :] / img_size[1] * 2) - 1
        # grid sample
        # pts [B, N, 1, 2] -> [B, C, N, 1]
        uv = rearrange(pts_2d[:, :2, :], 'b (t c) n -> b n t c', t=1)
        # print(uv)
        samples = torch.nn.functional.grid_sample(feature, uv, align_corners=True, mode='bilinear')
        pts_feature = rearrange(samples, 'b c n t -> b c (n t)')
        if rt_pts_2d:
            return pts_feature, pts_2d
        return pts_feature

    def query_model(self, feature, pts):
        xyz = rearrange(pts, 'b (t1 t2 c) n -> b n t1 t2 c', t1=1, t2=1)
        samples = torch.nn.functional.grid_sample(feature, xyz, align_corners=True, mode='bilinear')
        pts_feature = rearrange(samples, 'b c n t1 t2 -> b c (n t1 t2)')
        return pts_feature