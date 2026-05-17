"""
Plantar pressure estimation model.
Predicts a per-region pressure distribution from global shape features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureNet(nn.Module):
    """
    Regression head that produces a pressure-distribution vector.

    Input:  (B, input_dim)   – global foot features (default 256)
    Output: (B, num_regions) – predicted normalised pressure per region
                               (default 10 plantar regions)
    """

    NUM_REGIONS = 10  # hallux, lesser toes, met1-5, midfoot, medial heel, lateral heel

    def __init__(self, input_dim: int = 256, num_regions: int = 10):
        super().__init__()
        self.num_regions = num_regions
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_regions)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # Sigmoid to bound predictions in [0, 1]
        return torch.sigmoid(x)

    def predict(self, x: torch.Tensor):
        """
        Returns:
            pressure_map: (num_regions,) numpy array in [0, 1]
            avg_pressure: float – mean pressure score
        """
        self.eval()
        with torch.no_grad():
            pmap = self.forward(x).squeeze().cpu().numpy()
        return pmap, float(pmap.mean())
