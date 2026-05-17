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
    # FIX: If too few points, skip segmentation entirely
    if len(points) < 4:
        log.warning("Only %d points — skipping segmentation", len(points))
        return points

    # FIX: Adapt min_samples to actual point count.
    # Original used DBSCAN_MIN_POINTS=100 which crashes on small meshes.
    effective_min = min(min_samples, max(2, len(points) // 4))
    log.info(
        "Running DBSCAN (eps=%.1f mm, min_samples=%d [configured=%d]) on %d points...",
        eps, effective_min, min_samples, len(points),
    )

    clustering = DBSCAN(eps=eps, min_samples=effective_min, n_jobs=-1)
    labels = clustering.fit_predict(points)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        log.warning("DBSCAN found no clusters — returning full point cloud")
        return points

    log.info("Found %d clusters", len(unique_labels))

    largest_label = max(unique_labels, key=lambda l: (labels == l).sum())
    foot_mask = labels == largest_label
    foot_pts = points[foot_mask]

    # FIX: If largest cluster is too small, return all points
    if len(foot_pts) < 4:
        log.warning("Largest cluster only %d pts — using full cloud", len(foot_pts))
        return points

    log.info(
        "Selected cluster %d with %d points (%.1f%% of total)",
        largest_label, foot_mask.sum(),
        100.0 * foot_mask.sum() / len(labels),
    )
    return foot_pts


def refine_segmentation(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Post-segmentation cleanup via statistical outlier removal."""
    # FIX: Skip if not enough points
    if len(points) < 6:
        log.warning("Too few points (%d) for refinement — skipping", len(points))
        return points

    k = min(nb_neighbors + 1, len(points))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    mean_dists = distances[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    if global_std < 1e-10:
        return points
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    removed = (~mask).sum()

    # FIX: Don't refine away too many points
    if mask.sum() < 4:
        log.warning("Refinement would leave <4 points — skipping")
        return points

    log.info("Refinement removed %d stray points", removed)
    return points[mask]
