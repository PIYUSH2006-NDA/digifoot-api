"""
Low-level geometry helpers used across multiple services.
All coordinates are expected in millimetres unless noted otherwise.
"""

import numpy as np
from typing import Tuple


def bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_corner, max_corner) of an axis-aligned bounding box."""
    return points.min(axis=0), points.max(axis=0)


def oriented_bounding_box_dims(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute length, width, height of the point cloud using PCA-aligned axes.
    Returns dimensions sorted descending (length >= width >= height).
    """
    if len(points) < 2:
        return (0.0, 0.0, 0.0)

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # FIX: Handle degenerate covariance (NaN, Inf, singular)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        dims = points.max(axis=0) - points.min(axis=0)
        return tuple(sorted(dims, reverse=True))

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        dims = points.max(axis=0) - points.min(axis=0)
        return tuple(sorted(dims, reverse=True))

    projected = centered @ eigenvectors
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    dims = maxs - mins
    return tuple(sorted(dims, reverse=True))


def foot_length_width(points: np.ndarray) -> Tuple[float, float]:
    """
    Estimate foot length (longest axis) and width (second axis).
    Uses PCA so orientation does not matter.
    """
    dims = oriented_bounding_box_dims(points)
    return float(dims[0]), float(dims[1])


def compute_arch_height(points: np.ndarray) -> float:
    """
    Estimate arch height as the vertical distance between the lowest
    point of the medial arch region and the ground plane.
    """
    if len(points) < 10:
        return 15.0  # Default normal arch height (mm)

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return 15.0

    try:
        _, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 15.0

    primary = eigvecs[:, -1]
    proj = centered @ primary
    p_min, p_max = proj.min(), proj.max()
    p_range = p_max - p_min

    if p_range < 1e-6:
        return 15.0

    third = p_range / 3.0
    mask = (proj >= p_min + third) & (proj <= p_max - third)
    middle_pts = points[mask]

    if len(middle_pts) < 3:
        return 15.0

    z_vals = middle_pts[:, 2]
    return float(z_vals.max() - z_vals.min())


def sample_points_uniform(points: np.ndarray, n: int) -> np.ndarray:
    """Uniformly resample a point cloud to exactly *n* points."""
    num = points.shape[0]
    if num == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if num >= n:
        indices = np.random.choice(num, n, replace=False)
    else:
        indices = np.random.choice(num, n, replace=True)
    return points[indices].astype(np.float32)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Centre to origin and scale to unit sphere."""
    centroid = points.mean(axis=0)
    pts = points - centroid
    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist > 1e-10:
        pts /= max_dist
    return pts.astype(np.float32)


def compute_normals_from_points(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate normals via local PCA on k-nearest neighbours.
    """
    from scipy.spatial import cKDTree

    if len(points) < 2:
        normals = np.zeros_like(points)
        if len(normals) > 0:
            normals[:, 2] = 1.0
        return normals.astype(np.float32)

    # FIX: Clamp k to point count
    effective_k = min(k, len(points))
    tree = cKDTree(points)
    normals = np.zeros_like(points)

    for i, pt in enumerate(points):
        _, idx = tree.query(pt, k=effective_k)
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        neighbours = points[idx]
        if len(neighbours) < 3:
            normals[i] = [0, 0, 1]
            continue
        cov = np.cov(neighbours, rowvar=False)
        if np.any(np.isnan(cov)):
            normals[i] = [0, 0, 1]
            continue
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]

    # Consistent orientation: flip normals pointing downward
    down = np.array([0, 0, -1.0])
    flip_mask = (normals @ down) > 0
    normals[flip_mask] *= -1
    return normals.astype(np.float32)
