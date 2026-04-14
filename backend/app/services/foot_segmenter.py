"""
Foot segmentation service.
Isolates the foot from background clutter using DBSCAN clustering.
Uses sklearn instead of Open3D for compatibility.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from app.config import DBSCAN_EPS, DBSCAN_MIN_POINTS
from app.utils.logger import get_logger

log = get_logger(__name__)


def segment_foot(
    points: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_POINTS,
) -> np.ndarray:
    """
    Segment the foot from the scene:
    1. Run DBSCAN clustering.
    2. Select the largest cluster (assumed to be the foot).
    3. Return points belonging to the foot cluster.
    """
    log.info("Running DBSCAN (eps=%.1f mm, min_samples=%d) ...", eps, min_samples)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(points)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    if not unique_labels:
        log.warning("DBSCAN found no clusters - returning full point cloud")
        return points

    log.info("Found %d clusters", len(unique_labels))

    # Pick the largest cluster
    largest_label = max(unique_labels, key=lambda l: (labels == l).sum())
    foot_mask = labels == largest_label
    foot_pts = points[foot_mask]

    log.info(
        "Selected cluster %d with %d points (%.1f%% of total)",
        largest_label,
        foot_mask.sum(),
        100.0 * foot_mask.sum() / len(labels),
    )
    return foot_pts


def refine_segmentation(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Post-segmentation cleanup via statistical outlier removal.
    """
    nn = NearestNeighbors(n_neighbors=min(nb_neighbors + 1, len(points)))
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    mean_dists = distances[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    removed = (~mask).sum()
    log.info("Refinement removed %d stray points", removed)
    return points[mask]
