"""
backend/app/ml/yolo_seg_model.py

Wrapper for YOLOv8-seg model used by the depth pipeline.
Mirrors the singleton pattern of model_loader.py for consistency.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class YOLOSegLoader:
    """
    Singleton wrapper around the ultralytics YOLO class.
    Falls back to None if no weights are available.
    """

    _instance: Optional["YOLOSegLoader"] = None

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = None
        self._load()

    def _load(self):
        if not Path(self.weights_path).exists():
            logger.warning(f"YOLO weights not found at {self.weights_path}")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            logger.info(f"✓ YOLO model loaded: {self.weights_path}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO: {e}")
            self.model = None

    @classmethod
    def get(cls, weights_path: Optional[str] = None) -> "YOLOSegLoader":
        if cls._instance is None:
            if weights_path is None:
                weights_path = os.getenv(
                    "YOLO_MODEL_PATH",
                    str(Path("weights") / "foot_yolov8_seg.pt"),
                )
            cls._instance = cls(weights_path)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image, conf: float = 0.5, iou: float = 0.45):
        if not self.is_loaded:
            return None
        return self.model(image, conf=conf, iou=iou, verbose=False)
