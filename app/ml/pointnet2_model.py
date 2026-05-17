"""
backend/app/ml/pointnet2_model.py
PointNet++ refinement net for foot mesh denoising / completion.

Heads:
  - displacement (3D vector per point, max ±5cm)
  - normal       (3D unit vector per point)

Inference:
  refined_pts = scan_pts + displacement
  refined_nrm = normal head output (unit vectors)

Uses pytorch3d if available; otherwise naive torch knn (slower but works).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch3d.ops import knn_points, knn_gather
    _HAS_P3D = True
except Exception:
    _HAS_P3D = False


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """x: [B,N,3] -> idx: [B,N,k]"""
    if _HAS_P3D:
        return knn_points(x, x, K=k).idx
    B, N, _ = x.shape
    d = torch.cdist(x, x)
    return d.topk(k, largest=False).indices


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x:[B,N,C], idx:[B,N,k] -> [B,N,k,C]"""
    if _HAS_P3D:
        return knn_gather(x, idx)
    B, N, C = x.shape
    k = idx.shape[-1]
    return x.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, C)).reshape(B, N, k, C)


class SetAbstract(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 32):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c + 3, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 1),    nn.BatchNorm2d(out_c), nn.ReLU(True),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        idx = _knn(xyz, self.k)                            # [B,N,k]
        nbr_xyz = _gather(xyz, idx)                        # [B,N,k,3]
        nbr_feat = _gather(feats, idx)                     # [B,N,k,C]
        rel = nbr_xyz - xyz.unsqueeze(2)                   # local frame
        x = torch.cat([rel, nbr_feat], dim=-1)             # [B,N,k,3+C]
        x = x.permute(0, 3, 1, 2)                          # [B,3+C,N,k]
        x = self.mlp(x).max(dim=-1).values                 # [B,out,N]
        return x.permute(0, 2, 1)                          # [B,N,out]


class PointNet2FootRefine(nn.Module):
    """
    Inference contract:
      xyz: [B, N, 3] float32, normalized to unit ball
    Returns:
      disp: [B, N, 3] float32, ±0.05
      nrm:  [B, N, 3] float32, unit vectors
    """
    def __init__(self):
        super().__init__()
        self.sa1 = SetAbstract(3,   64,  k=32)
        self.sa2 = SetAbstract(64,  128, k=32)
        self.sa3 = SetAbstract(128, 256, k=32)
        head_in = 256 + 128 + 64 + 3
        self.head_disp = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(True),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 3),
        )
        self.head_norm = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(True),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 3),
        )

    def forward(self, xyz: torch.Tensor):
        f0 = xyz
        f1 = self.sa1(xyz, f0)
        f2 = self.sa2(xyz, f1)
        f3 = self.sa3(xyz, f2)
        cat = torch.cat([xyz, f1, f2, f3], dim=-1)
        disp = torch.tanh(self.head_disp(cat)) * 0.05      # ±5cm
        nrm = F.normalize(self.head_norm(cat), dim=-1)
        return disp, nrm
