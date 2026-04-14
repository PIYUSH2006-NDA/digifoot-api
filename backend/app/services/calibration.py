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
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    dims = bb_max - bb_min
    length = dims.max()  # longest axis ≈ foot length

    log.info(
        "Bounding box dimensions (mm): %.1f x %.1f x %.1f  (max=%.1f)",
        dims[0], dims[1], dims[2], length,
    )

    if MIN_FOOT_LENGTH_MM <= length <= MAX_FOOT_LENGTH_MM:
        log.info("Dimensions within plausible foot range [OK]")
        return True

    log.warning(
        "Dimensions OUTSIDE plausible range [%.0f, %.0f] mm - got %.1f mm",
        MIN_FOOT_LENGTH_MM,
        MAX_FOOT_LENGTH_MM,
        length,
    )
    return False


def auto_calibrate(points: np.ndarray) -> np.ndarray:
    """
    Attempt automatic calibration.
    1. If largest dimension < 1 → assume metres, multiply by 1000.
    2. If largest dimension is already in a plausible foot range → keep as-is.
    3. Otherwise apply default SCALE_FACTOR.
    """
    max_dim = (points.max(axis=0) - points.min(axis=0)).max()

    if max_dim < 1.0:
        log.info("Max dimension %.4f -> assuming metres, converting to mm", max_dim)
        return apply_scale(points, 1000.0)

    if MIN_FOOT_LENGTH_MM <= max_dim <= MAX_FOOT_LENGTH_MM:
        log.info("Max dimension %.1f mm -> already calibrated", max_dim)
        return points

    log.info("Max dimension %.2f -> applying default scale factor", max_dim)
    return apply_scale(points, SCALE_FACTOR)
