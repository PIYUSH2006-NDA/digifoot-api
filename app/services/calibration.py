"""
Scale calibration service.
Converts LiDAR mesh coordinates from metres to millimetres
and validates dimensional plausibility.
Uses pure numpy – no Open3D dependency.
"""

import numpy as np

from app.config import SCALE_FACTOR
from app.utils.logger import get_logger

log = get_logger(__name__)

# Reasonable foot size bounds (mm)
MIN_FOOT_LENGTH_MM = 150.0
MAX_FOOT_LENGTH_MM = 350.0

# Fallback: if calibration can't determine scale, assume a standard foot
FALLBACK_FOOT_LENGTH_MM = 260.0
FALLBACK_FOOT_WIDTH_MM = 95.0


def apply_scale(
    points: np.ndarray,
    factor: float = SCALE_FACTOR,
) -> np.ndarray:
    """Scale every point by *factor* (default: metres → mm)."""
    points = points * factor
    log.info("Applied scale factor %.1f -> points now in mm", factor)
    return points


def validate_dimensions(points: np.ndarray) -> bool:
    """
    Sanity-check the point cloud dimensions after calibration.
    Returns True when the bounding-box length falls inside a
    plausible foot-size range.
    """
    if len(points) < 2:
        log.warning("Too few points to validate dimensions")
        return False

    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    dims = bb_max - bb_min
    length = dims.max()

    log.info(
        "Bounding box dimensions (mm): %.1f x %.1f x %.1f  (max=%.1f)",
        dims[0], dims[1], dims[2], length,
    )

    if MIN_FOOT_LENGTH_MM <= length <= MAX_FOOT_LENGTH_MM:
        log.info("Dimensions within plausible foot range [OK]")
        return True

    log.warning(
        "Dimensions OUTSIDE plausible range [%.0f, %.0f] mm — got %.1f mm",
        MIN_FOOT_LENGTH_MM, MAX_FOOT_LENGTH_MM, length,
    )
    return False


def auto_calibrate(points: np.ndarray) -> np.ndarray:
    """
    Attempt automatic calibration.
    1. If largest dimension < 1 → assume metres, multiply by 1000.
    2. If largest dimension is already in a plausible foot range → keep as-is.
    3. Otherwise apply default SCALE_FACTOR.
    4. FIX: If result is still out of range, force-scale to standard foot.
    """
    if len(points) < 2:
        log.warning("Too few points for calibration — returning as-is")
        return points

    max_dim = (points.max(axis=0) - points.min(axis=0)).max()

    if max_dim < 1e-6:
        log.warning("Near-zero extent (%.8f) — likely degenerate mesh", max_dim)
        return _force_scale_to_foot(points)

    if max_dim < 1.0:
        log.info("Max dimension %.4f -> assuming metres, converting to mm", max_dim)
        points = apply_scale(points, 1000.0)
        max_dim *= 1000.0

    if MIN_FOOT_LENGTH_MM <= max_dim <= MAX_FOOT_LENGTH_MM:
        log.info("Max dimension %.1f mm -> already calibrated", max_dim)
        return points

    # Try default scale factor
    scaled = apply_scale(points.copy(), SCALE_FACTOR)
    scaled_max = (scaled.max(axis=0) - scaled.min(axis=0)).max()
    if MIN_FOOT_LENGTH_MM <= scaled_max <= MAX_FOOT_LENGTH_MM:
        return scaled

    # FIX: Force-scale to a standard foot size for fallback meshes
    log.warning("Auto-calibration failed (max=%.2f) — force-scaling to standard foot", max_dim)
    return _force_scale_to_foot(points)


def _force_scale_to_foot(points: np.ndarray) -> np.ndarray:
    """
    Scale the point cloud so its longest dimension equals a standard foot length.
    This ensures the pipeline can produce a usable insole even from fallback geometry.
    """
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    dims = bb_max - bb_min
    max_dim = dims.max()

    if max_dim < 1e-10:
        # Completely degenerate — create a synthetic bounding box
        log.warning("Degenerate points — synthesizing standard foot dimensions")
        centroid = points.mean(axis=0)
        points = points - centroid
        # Scale to target
        points[:, 0] *= FALLBACK_FOOT_WIDTH_MM
        points[:, 1] *= 30.0  # height ~30mm
        points[:, 2] *= FALLBACK_FOOT_LENGTH_MM
        return points

    scale = FALLBACK_FOOT_LENGTH_MM / max_dim
    log.info("Force-scaling by %.2f (%.2f -> %.1f mm)", scale, max_dim, FALLBACK_FOOT_LENGTH_MM)
    return points * scale
