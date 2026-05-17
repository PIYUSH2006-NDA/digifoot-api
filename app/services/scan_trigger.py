# scan_trigger.py
# Real-time scan quality validation + auto-capture trigger
# Validates: presence, centering, distance, orientation, temporal stability

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ScanState:
    is_foot_present: bool = False
    is_centered: bool = False
    is_correct_distance: bool = False
    is_stable: bool = False
    is_oriented: bool = False
    is_coverage_ok: bool = False
    confidence: float = 0.0
    frames_stable: int = 0
    ready_to_capture: bool = False
    mean_depth: float = 0.0
    centroid: Optional[Tuple[int, int]] = None


class ScanTrigger:
    """
    Frame-by-frame scan quality gate.
    Call .update() on every ARFrame. When state.ready_to_capture == True,
    trigger capture and pass to FootReconstructor.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        target_dist_min: float = 0.35,      # meters — minimum scan distance
        target_dist_max: float = 0.80,      # meters — maximum scan distance
        center_tolerance: float = 0.20,     # fraction of frame (0–1)
        min_coverage_frac: float = 0.05,    # foot must cover ≥5% of frame
        max_coverage_frac: float = 0.45,    # foot must cover ≤45% of frame
        stable_frames_required: int = 10,   # consecutive stable frames to capture
        stability_depth_std: float = 0.010, # max depth std dev for stability (meters)
        mask_iou_threshold: float = 0.85,   # min mask IoU between consecutive frames
    ):
        self.fw = frame_width
        self.fh = frame_height
        self.dist_min = target_dist_min
        self.dist_max = target_dist_max
        self.center_tol = center_tolerance
        self.min_cov = min_coverage_frac
        self.max_cov = max_coverage_frac
        self.stable_req = stable_frames_required
        self.stable_thresh = stability_depth_std
        self.iou_thresh = mask_iou_threshold

        # Temporal buffers
        self.depth_buffer: deque = deque(maxlen=stable_frames_required)
        self.mask_buffer: deque = deque(maxlen=5)

        self.state = ScanState()

    # ------------------------------------------------------------------ #
    #  Per-check methods
    # ------------------------------------------------------------------ #

    def _get_centroid(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        M = cv2.moments(mask)
        if M["m00"] < 100:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def _check_centered(self, centroid: Tuple[int, int]) -> bool:
        fcx, fcy = self.fw // 2, self.fh // 2
        dx = abs(centroid[0] - fcx) / self.fw
        dy = abs(centroid[1] - fcy) / self.fh
        return dx < self.center_tol and dy < self.center_tol

    def _check_distance(self, depth: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
        foot_depth = depth.copy()
        foot_depth[mask < 128] = np.nan
        valid = foot_depth[~np.isnan(foot_depth)]
        if len(valid) < 100:
            return False, 0.0
        med = float(np.median(valid))
        return self.dist_min <= med <= self.dist_max, med

    def _check_orientation(self, mask: np.ndarray) -> bool:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return False
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            return False
        ratio = max(w, h) / min(w, h)
        return 1.8 <= ratio <= 4.2

    def _check_coverage(self, mask: np.ndarray) -> bool:
        frac = (mask > 128).sum() / (self.fw * self.fh)
        return self.min_cov <= frac <= self.max_cov

    def _check_temporal_stability(self, depth: np.ndarray) -> bool:
        """Depth std dev across frame buffer must be below threshold."""
        self.depth_buffer.append(np.nan_to_num(depth, nan=0.0))
        if len(self.depth_buffer) < self.stable_req:
            return False
        stack = np.stack(list(self.depth_buffer), axis=0)
        temporal_std = np.std(stack, axis=0)
        active = temporal_std[temporal_std > 0]
        if active.size == 0:
            return False
        return float(active.mean()) < self.stable_thresh

    def _check_mask_iou(self, mask: np.ndarray) -> bool:
        """IoU between current and previous mask must exceed threshold."""
        self.mask_buffer.append(mask.copy())
        if len(self.mask_buffer) < 2:
            return False
        prev = self.mask_buffer[-2].astype(bool)
        curr = self.mask_buffer[-1].astype(bool)
        inter = (prev & curr).sum()
        union = (prev | curr).sum()
        if union == 0:
            return False
        return (inter / union) >= self.iou_thresh

    # ------------------------------------------------------------------ #
    #  Main update
    # ------------------------------------------------------------------ #

    def update(
        self,
        depth: np.ndarray,
        mask: Optional[np.ndarray],
        conf: float = 1.0,
    ) -> ScanState:
        """
        Update scan state from current frame.
        Call every ARFrame. Returns current ScanState.

        Args:
            depth: preprocessed float32 depth map (meters, NaN for invalid)
            mask:  uint8 binary segmentation mask (0 or 255)
            conf:  YOLO detection confidence (0–1)
        """
        if mask is None or mask.sum() < 100:
            self.state = ScanState()
            return self.state

        centroid = self._get_centroid(mask)
        dist_ok, mean_d = self._check_distance(depth, mask)

        self.state.is_foot_present = conf > 0.35 or mask.sum() > 4000
        self.state.is_centered = self._check_centered(centroid) if centroid else False
        self.state.is_correct_distance = dist_ok
        self.state.is_oriented = self._check_orientation(mask)
        self.state.is_coverage_ok = self._check_coverage(mask)
        self.state.confidence = conf
        self.state.mean_depth = mean_d
        self.state.centroid = centroid

        stable_depth = self._check_temporal_stability(depth)
        stable_mask = self._check_mask_iou(mask)
        self.state.is_stable = stable_depth and stable_mask

        if self.state.is_stable:
            self.state.frames_stable += 1
        else:
            self.state.frames_stable = 0

        self.state.ready_to_capture = (
            self.state.is_foot_present
            and self.state.is_centered
            and self.state.is_correct_distance
            and self.state.is_oriented
            and self.state.is_coverage_ok
            and self.state.frames_stable >= self.stable_req
        )

        return self.state

    def get_guidance(self) -> str:
        """Human-readable guidance string for UI overlay."""
        s = self.state
        if not s.is_foot_present:
            return "Place foot in frame"
        if not s.is_coverage_ok:
            return "Move phone closer" if (s.mean_depth > self.dist_max) else "Move phone farther"
        if not s.is_correct_distance:
            return "Adjust distance (35–80cm)"
        if not s.is_centered:
            return "Center foot in frame"
        if not s.is_oriented:
            return "Rotate — point toes away from you"
        if not s.is_stable:
            remaining = self.stable_req - s.frames_stable
            return f"Hold still... ({remaining} frames)"
        if s.ready_to_capture:
            return "✓ Capturing!"
        return "Hold steady..."

    def get_progress(self) -> float:
        """Capture readiness 0.0–1.0 for progress bar UI."""
        checks = [
            self.state.is_foot_present,
            self.state.is_coverage_ok,
            self.state.is_correct_distance,
            self.state.is_centered,
            self.state.is_oriented,
        ]
        base = sum(checks) / len(checks) * 0.6
        stability_progress = min(self.state.frames_stable / self.stable_req, 1.0) * 0.4
        return round(base + stability_progress, 2)

    def reset(self):
        """Reset trigger state (call after successful capture)."""
        self.depth_buffer.clear()
        self.mask_buffer.clear()
        self.state = ScanState()
