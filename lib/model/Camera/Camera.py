import torch

class Camera():
    def __init__(self):
        pass

    def perspective(self, pts, calibs):
        # pts: [B, 3, N]
        # calibs: [B, 4, 4]
        pts = torch.bmm(calibs[:, :3, :3], pts)
        pts = pts + calibs[:, :3, 3:4]
        pts[:, :2, :] /= pts[:, 2:, :]
        pts[torch.isnan(pts)] = 0
        pts[torch.isinf(pts)] = 0
        return pts.detach()