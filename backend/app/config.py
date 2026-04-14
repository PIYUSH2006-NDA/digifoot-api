"""
Application configuration.
Central configuration management with environment variable overrides.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
SCANS_DIR = BASE_DIR / "scans"
STLS_DIR = BASE_DIR / "stls"
WEIGHTS_DIR = BASE_DIR / "weights"

# Ensure runtime directories exist
for _dir in (SCANS_DIR, STLS_DIR, WEIGHTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------
HOST: str = os.getenv("APP_HOST", "0.0.0.0")
PORT: int = int(os.getenv("APP_PORT", "8000"))
DEBUG: bool = os.getenv("APP_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Processing settings
# ---------------------------------------------------------------------------
# LiDAR mesh arrives in metres; we convert to mm for manufacturing.
SCALE_FACTOR: float = float(os.getenv("SCALE_FACTOR", "1000.0"))

# Mesh cleaning
VOXEL_DOWNSAMPLE_SIZE: float = float(os.getenv("VOXEL_DOWNSAMPLE_SIZE", "0.5"))  # mm
STATISTICAL_NB_NEIGHBORS: int = int(os.getenv("STAT_NB_NEIGHBORS", "30"))
STATISTICAL_STD_RATIO: float = float(os.getenv("STAT_STD_RATIO", "2.0"))

# DBSCAN segmentation
DBSCAN_EPS: float = float(os.getenv("DBSCAN_EPS", "5.0"))       # mm
DBSCAN_MIN_POINTS: int = int(os.getenv("DBSCAN_MIN_POINTS", "100"))

# Poisson reconstruction depth
POISSON_DEPTH: int = int(os.getenv("POISSON_DEPTH", "9"))

# ---------------------------------------------------------------------------
# ML settings
# ---------------------------------------------------------------------------
POINTNET_WEIGHTS: str = os.getenv(
    "POINTNET_WEIGHTS",
    str(WEIGHTS_DIR / "pointnet_foot.pth"),
)
ARCH_CLASSIFIER_WEIGHTS: str = os.getenv(
    "ARCH_CLASSIFIER_WEIGHTS",
    str(WEIGHTS_DIR / "arch_classifier.pth"),
)
PRESSURE_MODEL_WEIGHTS: str = os.getenv(
    "PRESSURE_MODEL_WEIGHTS",
    str(WEIGHTS_DIR / "pressure_model.pth"),
)
ML_NUM_POINTS: int = int(os.getenv("ML_NUM_POINTS", "2048"))

# ---------------------------------------------------------------------------
# Insole generation
# ---------------------------------------------------------------------------
INSOLE_THICKNESS_MM: float = float(os.getenv("INSOLE_THICKNESS", "3.0"))
INSOLE_ARCH_HEIGHT_FLAT: float = float(os.getenv("ARCH_HEIGHT_FLAT", "8.0"))
INSOLE_ARCH_HEIGHT_NORMAL: float = float(os.getenv("ARCH_HEIGHT_NORMAL", "15.0"))
INSOLE_ARCH_HEIGHT_HIGH: float = float(os.getenv("ARCH_HEIGHT_HIGH", "22.0"))
HEEL_CUP_DEPTH_MM: float = float(os.getenv("HEEL_CUP_DEPTH", "12.0"))
FOREFOOT_CUSHION_MM: float = float(os.getenv("FOREFOOT_CUSHION", "5.0"))

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_TITLE: str = "Orthopedic Insole Pipeline"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Production-ready backend for generating custom orthopedic insoles "
    "from LiDAR-based foot scans."
)
