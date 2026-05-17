"""
backend/app/config.py

Configuration — add these settings to your existing config.py.
Below is a complete config showing depth-pipeline-specific additions.
If you already have a config.py, merge the V2 / DEPTH_* / CAMERA_* settings into it.
"""

import os
from pathlib import Path


class Settings:
    # ── Existing (legacy) ─────────────────────────────────────────────
    SCALE_FACTOR: float = float(os.getenv("SCALE_FACTOR", 1000.0))
    VOXEL_DOWNSAMPLE_SIZE: float = float(os.getenv("VOXEL_DOWNSAMPLE_SIZE", 0.5))
    DBSCAN_EPS: float = float(os.getenv("DBSCAN_EPS", 5.0))
    DBSCAN_MIN_POINTS: int = int(os.getenv("DBSCAN_MIN_POINTS", 100))
    POISSON_DEPTH: int = int(os.getenv("POISSON_DEPTH", 9))
    ML_NUM_POINTS: int = int(os.getenv("ML_NUM_POINTS", 2048))
    INSOLE_THICKNESS: float = float(os.getenv("INSOLE_THICKNESS", 3.0))
    HEEL_CUP_DEPTH: float = float(os.getenv("HEEL_CUP_DEPTH", 12.0))
    ARCH_HEIGHT_FLAT: float = float(os.getenv("ARCH_HEIGHT_FLAT", 8.0))
    ARCH_HEIGHT_NORMAL: float = float(os.getenv("ARCH_HEIGHT_NORMAL", 15.0))
    ARCH_HEIGHT_HIGH: float = float(os.getenv("ARCH_HEIGHT_HIGH", 22.0))

    # ── Storage ──────────────────────────────────────────────────────
    WEIGHTS_DIR: str = os.getenv("WEIGHTS_DIR", "weights")
    SCANS_DIR: str = os.getenv("SCANS_DIR", "scans")
    STLS_DIR: str = os.getenv("STLS_DIR", "stls")
    OUTPUTS_DIR: str = os.getenv("OUTPUTS_DIR", "outputs")

    # ── V2 Depth pipeline settings ───────────────────────────────────
    # Camera intrinsics (default = iPhone TrueDepth)
    CAMERA_FX: float = float(os.getenv("CAMERA_FX", 585.0))
    CAMERA_FY: float = float(os.getenv("CAMERA_FY", 585.0))
    CAMERA_CX: float = float(os.getenv("CAMERA_CX", 256.0))
    CAMERA_CY: float = float(os.getenv("CAMERA_CY", 192.0))

    # Depth preprocessing
    DEPTH_SCALE: float = float(os.getenv("DEPTH_SCALE", 1000.0))     # uint16 → meters
    DEPTH_MIN_M: float = float(os.getenv("DEPTH_MIN_M", 0.20))
    DEPTH_MAX_M: float = float(os.getenv("DEPTH_MAX_M", 1.50))

    # Floor removal
    FLOOR_RANSAC_THRESHOLD: float = float(os.getenv("FLOOR_RANSAC_THRESHOLD", 0.02))
    FLOOR_MARGIN_ABOVE: float = float(os.getenv("FLOOR_MARGIN_ABOVE", 0.03))

    # YOLO model filename inside WEIGHTS_DIR (auto-loaded if present)
    YOLO_MODEL_NAME: str = os.getenv("YOLO_MODEL_NAME", "foot_yolov8_seg.pt")

    # Scan trigger
    STABLE_FRAMES_REQUIRED: int = int(os.getenv("STABLE_FRAMES_REQUIRED", 10))
    CENTER_TOLERANCE: float = float(os.getenv("CENTER_TOLERANCE", 0.20))

    # Reconstruction
    RECON_TARGET_TRIANGLES: int = int(os.getenv("RECON_TARGET_TRIANGLES", 50000))
    RECON_VOXEL_SIZE_M: float = float(os.getenv("RECON_VOXEL_SIZE_M", 0.002))


settings = Settings()
