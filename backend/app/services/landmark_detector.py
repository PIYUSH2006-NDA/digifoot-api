"""
Anatomical landmark detection.
Identifies heel centre, arch peak, forefoot region, and toe tips
from a foot point cloud.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from app.services.geometry_utils import foot_length_width, compute_arch_height
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FootLandmarks:
    heel_center: np.ndarray          # (3,)
    arch_peak: np.ndarray            # (3,)
    forefoot_center: np.ndarray      # (3,)
    toe_tips: List[np.ndarray]       # list of (3,)
    foot_length_mm: float
    foot_width_mm: float
    arch_height_mm: float


def detect_landmarks(points: np.ndarray) -> FootLandmarks:
    """
    Heuristic landmark detection using PCA-aligned axes.

    Assumes the foot is oriented roughly along the first principal component.
    The algorithm divides the foot into heel / arch / forefoot / toe regions
    by percentile along the primary axis.
    """
    log.info("Detecting landmarks on %d points", len(points))

    # --- PCA alignment ---
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Primary axis = largest eigenvalue (last column)
    primary_axis = eigvecs[:, -1]

    projections = centered @ primary_axis
    p_min, p_max = projections.min(), projections.max()
    p_range = p_max - p_min

    # --- Heel: bottom 10 % ---
    heel_mask = projections <= (p_min + 0.10 * p_range)
    heel_pts = points[heel_mask]
    heel_center = heel_pts.mean(axis=0) if len(heel_pts) > 0 else centroid.copy()

    # --- Arch: 30 %–50 % ---
    arch_mask = (projections >= p_min + 0.30 * p_range) & (
        projections <= p_min + 0.50 * p_range
    )
    arch_pts = points[arch_mask]
    if len(arch_pts) > 0:
        # Arch peak = highest Z in the arch region
        arch_peak = arch_pts[arch_pts[:, 2].argmax()]
    else:
        arch_peak = centroid.copy()

    # --- Forefoot: 60 %–80 % ---
    fore_mask = (projections >= p_min + 0.60 * p_range) & (
        projections <= p_min + 0.80 * p_range
    )
    fore_pts = points[fore_mask]
    forefoot_center = fore_pts.mean(axis=0) if len(fore_pts) > 0 else centroid.copy()

    # --- Toe tips: top 5 % ---
    toe_mask = projections >= (p_min + 0.95 * p_range)
    toe_pts = points[toe_mask]
    toe_tips: List[np.ndarray] = []
    if len(toe_pts) > 0:
        # Simple clustering by lateral axis (second eigenvector)
        lateral = eigvecs[:, -2]
        lat_proj = (toe_pts - centroid) @ lateral
        # Split into ~5 bins for individual toe approximation
        bins = np.linspace(lat_proj.min(), lat_proj.max(), 6)
        for i in range(5):
            bin_mask = (lat_proj >= bins[i]) & (lat_proj < bins[i + 1])
            bin_pts = toe_pts[bin_mask]
            if len(bin_pts) > 0:
                tip_idx = ((bin_pts - centroid) @ primary_axis).argmax()
                toe_tips.append(bin_pts[tip_idx])
    if not toe_tips:
        toe_tips.append(points[projections.argmax()])

    # --- Measurements ---
    length, width = foot_length_width(points)
    arch_h = compute_arch_height(points)

    log.info(
        "Landmarks detected - length=%.1f mm, width=%.1f mm, arch=%.1f mm",
        length, width, arch_h,
    )

    return FootLandmarks(
        heel_center=heel_center,
        arch_peak=arch_peak,
        forefoot_center=forefoot_center,
        toe_tips=toe_tips,
        foot_length_mm=length,
        foot_width_mm=width,
        arch_height_mm=arch_h,
    )
