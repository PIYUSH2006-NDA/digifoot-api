"""
Arch-type classifier.
Classifies a foot into flat / normal / high arch
using PointNet-extracted features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArchClassifier(nn.Module):
    """
    MLP head that takes a global feature vector (from PointNet or
    concatenation of PointNet embedding + hand-crafted features)
    and outputs arch-type logits for 3 classes.

    Input:  (B, input_dim)   – default 256
    Output: (B, 3)           – logits for [flat, normal, high]
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # raw logits; apply softmax externally

    def predict(self, x: torch.Tensor):
        """
        Convenience method.
        Returns:
            label: int (0=flat, 1=normal, 2=high)
            confidence: float
            probabilities: (3,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, label = probs.max(dim=-1)
        return label.item(), confidence.item(), probs.squeeze().cpu().numpy()
