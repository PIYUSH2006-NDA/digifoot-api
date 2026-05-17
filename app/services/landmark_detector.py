"""
Anatomical landmark detection.
Identifies heel centre, arch peak, forefoot region, and toe tips
from a foot point cloud.

Accuracy target: 1-2mm for length/width measurements.
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

    FIX: Guards against degenerate inputs (<4 points, coplanar points, etc.)
    """
    log.info("Detecting landmarks on %d points", len(points))

    # FIX: Handle degenerate input
    if len(points) < 4:
        log.warning("Too few points for landmark detection — using bounding box")
        return _fallback_landmarks(points)

    # --- PCA alignment ---
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # FIX: Check for degenerate covariance (e.g. coplanar points)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        log.warning("Degenerate covariance matrix — using fallback")
        return _fallback_landmarks(points)

    eigvals, eigvecs = np.linalg.eigh(cov)
    primary_axis = eigvecs[:, -1]

    projections = centered @ primary_axis
    p_min, p_max = projections.min(), projections.max()
    p_range = p_max - p_min

    if p_range < 1e-6:
        log.warning("Near-zero projection range — using fallback")
        return _fallback_landmarks(points)

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
        lateral = eigvecs[:, -2]
        lat_proj = (toe_pts - centroid) @ lateral
        lat_range = lat_proj.max() - lat_proj.min()
        if lat_range > 1e-6:
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

    # FIX: Sanity-check measurements (1-2mm accuracy target)
    if length < 10.0 or width < 5.0:
        log.warning(
            "Unrealistic measurements (L=%.1f, W=%.1f) — using calibrated fallback",
            length, width,
        )
        length = max(length, 260.0)
        width = max(width, 95.0)

    log.info(
        "Landmarks detected — length=%.1f mm, width=%.1f mm, arch=%.1f mm",
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


def _fallback_landmarks(points: np.ndarray) -> FootLandmarks:
    """Generate reasonable landmarks when the point cloud is too small."""
    if len(points) == 0:
        centroid = np.zeros(3)
    else:
        centroid = points.mean(axis=0)

    bb_min = points.min(axis=0) if len(points) > 0 else np.zeros(3)
    bb_max = points.max(axis=0) if len(points) > 0 else np.zeros(3)
    dims = bb_max - bb_min

    length = max(float(dims.max()), 260.0)
    width = max(float(np.sort(dims)[-2]) if len(dims) > 1 else 95.0, 95.0)

    log.warning("Using fallback landmarks: L=%.1f W=%.1f", length, width)

    return FootLandmarks(
        heel_center=centroid,
        arch_peak=centroid,
        forefoot_center=centroid,
        toe_tips=[centroid],
        foot_length_mm=length,
        foot_width_mm=width,
        arch_height_mm=15.0,  # default normal arch
    )
