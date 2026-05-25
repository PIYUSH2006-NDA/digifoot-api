# app/ml/loaded_yolo.py
#
# Lazy singleton loader for the loaded-mode YOLOv8-seg model.
#
# The reconstruction pipeline asks for this model to refine foot
# segmentation. If the weights file is absent it returns None and the
# pipeline falls back: stored on-device mask -> geometric segmentation.
# Same graceful-degradation principle as the rest of the v2 pipeline.

import os
from pathlib import Path

# Override with the FOOT_YOLO_WEIGHTS env var if the path differs.
WEIGHTS_PATH = os.environ.get(
    "FOOT_YOLO_WEIGHTS", "weights/foot_yolov8_seg.pt"
)

_model = None
_attempted = False


def get_loaded_yolo():
    """Return the YOLOv8-seg model, or None if weights are unavailable.

    Loaded once on first call and cached for the process lifetime.
    """
    global _model, _attempted
    if _attempted:
        return _model
    _attempted = True

    path = Path(WEIGHTS_PATH)
    if not path.exists():
        print(f"[loaded_yolo] weights not found at {path} — geometric fallback")
        _model = None
        return None

    try:
        from ultralytics import YOLO
        _model = YOLO(str(path))
        print(f"[loaded_yolo] loaded YOLOv8-seg from {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[loaded_yolo] failed to load model: {exc}")
        _model = None

    return _model