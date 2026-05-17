"""
Model loader utility.
Handles weight loading with graceful fallback to random initialisation
when pretrained weights are not available.
"""

from pathlib import Path

import torch

from app.ml.pointnet_model import PointNetFootModel
from app.ml.arch_classifier import ArchClassifier
from app.ml.pressure_model import PressureNet
from app.config import (
    POINTNET_WEIGHTS,
    ARCH_CLASSIFIER_WEIGHTS,
    PRESSURE_MODEL_WEIGHTS,
    ML_NUM_POINTS,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# Module-level singletons (loaded once, reused)
_pointnet: PointNetFootModel | None = None
_arch_clf: ArchClassifier | None = None
_pressure: PressureNet | None = None
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("ML device: %s", _device)
    return _device


def _try_load_weights(model: torch.nn.Module, path: str, name: str) -> bool:
    """Attempt to load weights; return True on success."""
    p = Path(path)
    if p.exists():
        try:
            state = torch.load(str(p), map_location=_get_device(), weights_only=True)
            model.load_state_dict(state)
            log.info("Loaded pretrained weights for %s from %s", name, p)
            return True
        except Exception as exc:
            log.warning("Failed to load weights for %s: %s - using random init", name, exc)
    else:
        log.warning("Weights file not found for %s at %s - using random init", name, p)
    return False


def get_pointnet() -> PointNetFootModel:
    """Return the singleton PointNet model."""
    global _pointnet
    if _pointnet is None:
        _pointnet = PointNetFootModel(feature_dim=256)
        _try_load_weights(_pointnet, POINTNET_WEIGHTS, "PointNet")
        _pointnet.to(_get_device())
        _pointnet.eval()
    return _pointnet


def get_arch_classifier() -> ArchClassifier:
    """Return the singleton ArchClassifier model."""
    global _arch_clf
    if _arch_clf is None:
        _arch_clf = ArchClassifier(input_dim=256)
        _try_load_weights(_arch_clf, ARCH_CLASSIFIER_WEIGHTS, "ArchClassifier")
        _arch_clf.to(_get_device())
        _arch_clf.eval()
    return _arch_clf


def get_pressure_model() -> PressureNet:
    """Return the singleton PressureNet model."""
    global _pressure
    if _pressure is None:
        _pressure = PressureNet(input_dim=256)
        _try_load_weights(_pressure, PRESSURE_MODEL_WEIGHTS, "PressureNet")
        _pressure.to(_get_device())
        _pressure.eval()
    return _pressure


def get_num_points() -> int:
    return ML_NUM_POINTS
