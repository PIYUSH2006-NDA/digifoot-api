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
    Returns dimensions sorted descending (length ≥ width ≥ height).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Project onto principal components
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

    Heuristic: take the middle third of the foot along the primary axis
    and measure height range.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)

    # Primary axis = eigenvector with largest eigenvalue (last column)
    primary = eigvecs[:, -1]
    proj = centered @ primary
    p_min, p_max = proj.min(), proj.max()
    third = (p_max - p_min) / 3.0

    # Middle third mask
    mask = (proj >= p_min + third) & (proj <= p_max - third)
    middle_pts = points[mask]

    if len(middle_pts) < 10:
        return 0.0

    # Arch height = range along the vertical (Z) axis in the middle section
    z_vals = middle_pts[:, 2]
    return float(z_vals.max() - z_vals.min())


def sample_points_uniform(points: np.ndarray, n: int) -> np.ndarray:
    """
    Uniformly resample a point cloud to exactly *n* points.
    If fewer points exist, upsample with replacement.
    """
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
    if max_dist > 0:
        pts /= max_dist
    return pts.astype(np.float32)


def compute_normals_from_points(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate normals via local PCA on k-nearest neighbours.
    Lightweight fallback when Open3D normal estimation isn't applicable.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    normals = np.zeros_like(points)
    for i, pt in enumerate(points):
        _, idx = tree.query(pt, k=min(k, len(points)))
        neighbours = points[idx]
        cov = np.cov(neighbours, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue → normal direction
    # Consistent orientation: flip normals pointing downward
    down = np.array([0, 0, -1.0])
    flip_mask = (normals @ down) > 0
    normals[flip_mask] *= -1
    return normals.astype(np.float32)
