"""
PointNet feature extractor for foot shape analysis.

Architecture:
  Input (N×3) → shared MLPs → max-pool → global feature vector (1024-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial Transformer Network for input/feature alignment."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialise as identity
        identity = (
            torch.eye(self.k, dtype=x.dtype, device=x.device)
            .view(1, self.k * self.k)
            .repeat(batch_size, 1)
        )
        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """
    PointNet encoder that produces a 1024-d global feature vector.
    Optionally returns per-point features for segmentation tasks.
    """

    def __init__(self, global_feat: bool = True):
        super().__init__()
        self.global_feat = global_feat
        self.stn = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, N) point cloud
        Returns:
            global_feat: (B, 1024) if self.global_feat else (B, 1088, N)
            transform: (B, 3, 3) spatial transform matrix
        """
        batch_size, _, n_pts = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)          # (B, N, 3)
        x = torch.bmm(x, trans)        # apply transform
        x = x.transpose(2, 1)          # (B, 3, N)

        point_feat = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(point_feat)))    # (B, 128, N)
        x = self.bn3(self.conv3(x))                     # (B, 1024, N)
        x = torch.max(x, 2)[0]                         # (B, 1024)

        if self.global_feat:
            return x, trans

        # Per-point feature: concat global + local
        x = x.unsqueeze(2).repeat(1, 1, n_pts)
        return torch.cat([x, point_feat], 1), trans


class PointNetFootModel(nn.Module):
    """
    Full PointNet model for foot shape feature extraction.
    Outputs a 256-d embedding useful for downstream tasks.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder = PointNetEncoder(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, feature_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, N)
        Returns:
            features: (B, feature_dim)
            transform: (B, 3, 3)
        """
        global_feat, trans = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, trans
