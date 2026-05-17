"""
backend/app/ml/pointnet_model.py
Lightweight PointNet for foot surface refinement.
Input:  B x 3 x N  (xyz)
Output: B x 3 x N  (refined xyz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFootModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 64, 1)
        self.c2 = nn.Conv1d(64, 128, 1)
        self.c3 = nn.Conv1d(128, 256, 1)
        self.c4 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.dc1 = nn.Conv1d(512 + 256 + 128 + 64, 256, 1)
        self.dc2 = nn.Conv1d(256, 128, 1)
        self.dc3 = nn.Conv1d(128, 64, 1)
        self.dc4 = nn.Conv1d(64, 3, 1)

        self.dbn1 = nn.BatchNorm1d(256)
        self.dbn2 = nn.BatchNorm1d(128)
        self.dbn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # x: B x 3 x N

        x_in = x

        f1 = F.relu(self.bn1(self.c1(x)))
        f2 = F.relu(self.bn2(self.c2(f1)))
        f3 = F.relu(self.bn3(self.c3(f2)))
        f4 = F.relu(self.bn4(self.c4(f3)))

        # Global max pooling
        g = torch.max(f4, dim=2, keepdim=True)[0]
        g = g.expand(-1, -1, x.size(2))

        # Feature concatenation
        cat = torch.cat([g, f3, f2, f1], dim=1)

        d1 = F.relu(self.dbn1(self.dc1(cat)))
        d2 = F.relu(self.dbn2(self.dc2(d1)))
        d3 = F.relu(self.dbn3(self.dc3(d2)))

        delta = self.dc4(d3)

        # Small residual refinement
        return x_in + delta * 0.01