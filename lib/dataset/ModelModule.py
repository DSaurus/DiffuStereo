import numpy as np
import trimesh
import torch
import math
from einops import rearrange, repeat
from lib.util.IO import save_samples_labels, readobj
from lib.util.Geometry import cross_3d, rotationY, rotationX, rotationZ

class ModelModule():
    def __init__(self, opt=None):
        if opt is not None:
            self.sampling_sigma = opt.sampling_sigma
            self.num_sample_inout = opt.num_sample_inout
    
    def sampling(self, mesh, b_min, b_max, show=False):
        radius_list = [self.sampling_sigma / 3, self.sampling_sigma, self.sampling_sigma * 2]
        surface_points = np.zeros((3 * self.num_sample_inout, 3))
        sample_points = np.zeros((3 * self.num_sample_inout, 3))
        for i in range(3):
            d = self.num_sample_inout
            surface_points[i * d:(i + 1) * d, :], _ = trimesh.sample.sample_surface(mesh, d)
            sample_points[i * d:(i + 1) * d, :] = surface_points[i * d:(i + 1) * d, :] + np.random.normal(
                scale=radius_list[i], size=(d, 3))

        # add random points within image space
        length = b_max - b_min
        random_points = np.random.rand(self.num_sample_inout, 3) * length.numpy() + b_min.numpy()
        sample_points = np.concatenate([sample_points, random_points], 0)
        inside = mesh.contains(sample_points)

        inside_points = sample_points[inside]
        np.random.shuffle(inside_points)
        outside_points = sample_points[np.logical_not(inside)]
        np.random.shuffle(outside_points)

        nin = inside_points.shape[0]
        inside_points = inside_points[:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                                :(self.num_sample_inout - nin)]
        # [3, N]
        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))],1)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        if show:
            save_samples_labels('show/samples.ply', samples.numpy().T, labels.numpy().T)
        return {
            'samples': samples,
            'labels': labels
        }

    def normalize_mesh(self, mesh):
        vertices = mesh.vertices
        b0 = np.min(vertices, axis=0)
        b1 = np.max(vertices, axis=0)
        center = (b0 + b1) / 2
        scale = np.min(1.0 / (b1 - b0)) * 0.9
        b_min = center - 0.5 / scale 
        b_max = center + 0.5 / scale

        return {
            'b_min': torch.FloatTensor(b_min),
            'b_max': torch.FloatTensor(b_max),
            'scale': torch.FloatTensor(scale),
            'center': torch.FloatTensor(center)
        }

    def multicam_normalize(self, pts, norm_para, rot):
        scale = rearrange(norm_para['scale'], '(b n c) -> b c n', c=1, n=1)
        center = rearrange(norm_para['center'], '(b n) c -> b c n', n=1)
        pts = (pts - center) * scale
        B, V, _, _ = rot.shape
        pts = repeat(pts, 'b c n -> b v c n', v=V)
        rot = rearrange(rot, 'b v h w -> (b v) h w')
        pts = rearrange(pts, 'b v c n -> (b v) c n')
        pts = torch.bmm(rot, pts)
        pts = rearrange(pts, '(b v) c n -> b v c n', b=B)
        return pts

    def cam_normalize(self, pts, norm_para, rot):
        scale = rearrange(norm_para['scale'], '(b n c) -> b c n', c=1, n=1)
        center = rearrange(norm_para['center'], '(b n) c -> b c n', n=1)
        pts = (pts - center) * scale
        pts = torch.bmm(rot, pts)
        return pts

    def normalize(self, pts, norm_para):
        scale = rearrange(norm_para['scale'], '(b n c) -> b c n', c=1, n=1)
        center = rearrange(norm_para['center'], '(b n) c -> b c n', n=1)
        pts = (pts - center) * scale
        return pts

class SMPLNormModule(ModelModule):
    def __init__(self, opt):
        super(SMPLNormModule, self).__init__(opt)
        self.norm_smpl_faces = readobj(opt.norm_smpl_faces)['f']
        self.random_rotation = self.opt.random_rotation

    def smpl_norm(self, smpl_path):
        smpl_mesh = readobj(smpl_path)
        b0 = np.min(smpl_mesh, axis=0)
        b1 = np.max(smpl_mesh, axis=0)
        center = (b0 + b1) / 2
        scale = np.min(1.0 / (b1 - b0)) * 0.9 * 2
        b_min = center - 1 / scale
        b_max = center + 1 / scale

        norm_dir = np.zeros((3))
        for f in self.smpl_faces:
            a, b, c = smpl_mesh[f[0]][0], smpl_mesh[f[1]][0], smpl_mesh[f[2]][0]
            norm_dir += cross_3d(c - a, b - a)
            
        x, z = norm_dir[0], norm_dir[2]
        theta = math.acos(z / math.sqrt(z * z + x * x))
        if x < 0:
            theta = 2 * math.acos(-1) - theta
        norm_rot = np.array(rotationY(-theta))

        if self.random_rotation:
            pi = math.acos(-1)
            beta = 40 * pi / 180
            rand_rot = np.array(rotationX((np.random.rand() - 0.5) * beta)) @ np.array(
                rotationY((np.random.rand() - 0.5) * beta)) @ np.array(rotationZ((np.random.rand() - 0.5) * beta))
            norm_rot = rand_rot @ norm_rot

        return {
            'b_min': torch.FloatTensor(b_min),
            'b_max': torch.FloatTensor(b_max),
            'scale': torch.FloatTensor(scale),
            'center': torch.FloatTensor(center),
            'norm_dir': torch.FloatTensor(norm_dir),
            'norm_rot': torch.FloatTensor(norm_rot)
        }

    def normalize(self, pts, norm_para):
        scale = norm_para['scale']
        center = norm_para['center']
        norm_rot = norm_para['norm_rot']

        pts = (pts - center) * scale
        pts = norm_rot @ pts
        return pts

    def normalize_and_vox(self, mesh_path, norm_para):
        scale = norm_para['scale']
        center = norm_para['center']
        norm_rot = norm_para['norm_rot']
        
        mesh = trimesh.load(mesh_path)

        translation = np.zeros((4, 4))
        translation[:3, 3] = -np.array(center) * scale.numpy()
        for i in range(3):
            translation[i, i] = scale.numpy()
        translation[3, 3] = 1
        mesh.apply_transform(translation)

        rotation = np.zeros((4, 4))
        rotation[3, 3] = 1
        rotation[:3, :3] = norm_rot
        mesh.apply_transform(rotation)

        vox = mesh.voxelized(pitch=1.0 / 128, method='binvox', bounds=np.array([[-1, -1, -1], [1, 1, 1]]),
                                 exact=True)
        vox.fill()
        return torch.FloatTensor(vox.matrix)


    