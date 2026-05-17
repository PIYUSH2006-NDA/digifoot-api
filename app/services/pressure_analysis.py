"""
Plantar pressure analysis service.
Uses PressureNet to estimate per-region pressure from shape features.
"""

from dataclasses import dataclass

import numpy as np
import torch

from app.ml.model_loader import get_pressure_model, _get_device
from app.utils.logger import get_logger

log = get_logger(__name__)

REGION_NAMES = [
    "hallux",
    "lesser_toes",
    "met_1",
    "met_2",
    "met_3",
    "met_4",
    "met_5",
    "midfoot",
    "medial_heel",
    "lateral_heel",
]


@dataclass
class PressureResult:
    region_pressures: dict       # region_name -> pressure (0-1)
    average_score: float         # mean pressure across all regions
    peak_region: str             # region with highest pressure
    peak_pressure: float


def run_pressure_analysis(features: np.ndarray) -> PressureResult:
    """
    Predict normalised pressure distribution from a 256-d shape embedding.
    """
    device = _get_device()
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    model = get_pressure_model()
    pmap, avg = model.predict(tensor)

    region_pressures = {name: float(pmap[i]) for i, name in enumerate(REGION_NAMES)}
    peak_idx = int(np.argmax(pmap))

    result = PressureResult(
        region_pressures=region_pressures,
        average_score=avg,
        peak_region=REGION_NAMES[peak_idx],
        peak_pressure=float(pmap[peak_idx]),
    )
    log.info(
        "Pressure analysis -> avg=%.3f  peak=%s (%.3f)",
        result.average_score,
        result.peak_region,
        result.peak_pressure,
    )
    return result
