"""
ML-based biomechanical analysis service.
Orchestrates PointNet feature extraction and arch classification.
"""

from dataclasses import dataclass

import numpy as np
import torch

from app.ml.model_loader import get_pointnet, get_arch_classifier, get_num_points, _get_device
from app.services.geometry_utils import sample_points_uniform, normalize_point_cloud
from app.utils.logger import get_logger

log = get_logger(__name__)

ARCH_LABELS = {0: "flat", 1: "normal", 2: "high"}


@dataclass
class BiomechanicsResult:
    features: np.ndarray         # (256,) global shape embedding
    arch_type: str               # "flat" | "normal" | "high"
    arch_label_id: int
    confidence: float
    class_probabilities: np.ndarray  # (3,)


def run_biomechanical_analysis(points: np.ndarray) -> BiomechanicsResult:
    """
    Full biomechanical analysis pipeline:
    1. Resample and normalise the point cloud.
    2. Extract global shape features with PointNet.
    3. Classify arch type.
    """
    n_pts = get_num_points()
    device = _get_device()

    # Prepare input tensor (B, 3, N)
    sampled = sample_points_uniform(points, n_pts)
    normalised = normalize_point_cloud(sampled)
    tensor = (
        torch.from_numpy(normalised)
        .float()
        .unsqueeze(0)          # (1, N, 3)
        .permute(0, 2, 1)     # (1, 3, N)
        .to(device)
    )

    # Feature extraction
    pointnet = get_pointnet()
    with torch.no_grad():
        features, _ = pointnet(tensor)    # (1, 256)

    # Arch classification
    arch_clf = get_arch_classifier()
    label_id, confidence, probs = arch_clf.predict(features)

    result = BiomechanicsResult(
        features=features.squeeze().cpu().numpy(),
        arch_type=ARCH_LABELS[label_id],
        arch_label_id=label_id,
        confidence=confidence,
        class_probabilities=probs,
    )
    log.info(
        "Biomechanics -> arch=%s  conf=%.3f  probs=%s",
        result.arch_type, result.confidence,
        np.array2string(probs, precision=3),
    )
    return result
