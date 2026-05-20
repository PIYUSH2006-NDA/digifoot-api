"""
backend/app/services/depth_pipeline.py

Depth-only foot scan pipeline orchestrator.
Used by the v2_scan route — separate from the legacy mesh-based pipeline.py.

Stages:
  1. Load depth frames from job directory
  2. Preprocess depth (filter, fill holes, normalize)
  3. Hybrid segmentation (geometric + YOLOv8-seg if model available)
  4. Foot isolation
  5. Multi-frame fusion (if multiple frames available)
  6. 3D reconstruction (Poisson mesh)
  7. Foot measurements
  8. STL export to stls/{job_id}.stl
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
import open3d as o3d

from .depth_preprocessing import DepthPreprocessor, FloorRemover
from .foot_segmentation import (
    FootDepthSegmenter,
    AdvancedFloorRemover,
    FootScanInference,
)
from ..recon.reconstruction_3d import FootReconstructor

logger = logging.getLogger(__name__)


# ======================================================================
#  PIPELINE
# ======================================================================

class DepthFootPipeline:
    """
    Depth-only foot scanning pipeline.
    Compatible with your existing job-store pattern (scans/{job_id}/, stls/{job_id}.stl).
    """

    def __init__(
        self,
        weights_dir: str = "weights",
        yolo_model_name: str = "foot_yolov8_seg.pt",
        fx: float = 585.0,
        fy: float = 585.0,
        cx: float = 320.0,   # FIX: was 256.0 — correct for 640×480 TrueDepth (w/2)
        cy: float = 240.0,   # FIX: was 192.0 — correct for 640×480 TrueDepth (h/2)
    ):
        self.prep = DepthPreprocessor(fx=fx, fy=fy, cx=cx, cy=cy)
        self.floor = FloorRemover(self.prep)
        self.geo_seg = FootDepthSegmenter(self.prep, self.floor)
        self.recon = FootReconstructor(self.prep)

        # Optional YOLO model
        yolo_path = Path(weights_dir) / yolo_model_name
        self.inference: Optional[FootScanInference] = None
        if yolo_path.exists():
            try:
                self.inference = FootScanInference(str(yolo_path), self.prep, self.geo_seg)
                logger.info(f"✓ YOLO model loaded: {yolo_path}")
            except Exception as e:
                logger.warning(f"⚠ YOLO load failed ({e}) — using geometric-only mode")
        else:
            logger.info("ℹ No YOLO weights found — running geometric-only mode")

    # ------------------------------------------------------------------ #
    #  Input loading
    # ------------------------------------------------------------------ #

    def load_depth_frames(self, scan_dir: str) -> List[np.ndarray]:
        """
        Load all depth frames from a scan directory.
        Supports: .png (16-bit), .tiff, .npy, .bin (raw float32 from iOS bin mode)
        """
        scan_dir = Path(scan_dir)
        frames = []

        # FIX: added .bin pattern — TrueDepthScanScreen uses capture_v7 channel
        # which writes raw float32 .bin files, not PNGs. Backend was only loading
        # PNGs → "No frames found" → processing always failed for this scan path.
        patterns = ["depth_*.png", "*.depth.png", "frame_*.png", "*.tiff", "*.npy",
                    "depth_*.bin"]  # ← NEW: raw float32 from iOS bin mode
        files = []
        for p in patterns:
            files.extend(scan_dir.rglob(p))

        files = sorted(set(files))

        for f in files:
            try:
                if f.suffix == ".npy":
                    arr = np.load(f)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32) / self.prep.depth_scale
                    frames.append(arr)

                elif f.suffix == ".bin":
                    # Raw float32 depth. Matching .txt holds "{w},{h},float32"
                    txt = f.with_suffix(".txt")
                    if txt.exists():
                        dims = txt.read_text().strip().split(",")
                        w_d, h_d = int(dims[0]), int(dims[1])
                        arr = np.frombuffer(f.read_bytes(), dtype=np.float32).reshape(h_d, w_d)
                        # Values already in meters (float32) — no scale conversion needed
                        frames.append(arr)
                    else:
                        logger.warning(f"No .txt dims for {f.name} — skipping")

                else:
                    depth = self.prep.load_depth(str(f))
                    frames.append(depth)
            except Exception as e:
                logger.warning(f"Skip {f}: {e}")

        logger.info(f"Loaded {len(frames)} depth frames from {scan_dir}")
        return frames

    # ------------------------------------------------------------------ #
    #  Per-frame processing
    # ------------------------------------------------------------------ #

    def process_single_frame(self, depth: np.ndarray) -> dict:
        """
        Process one depth frame → isolated foot depth.
        """
        # Clean
        depth = self.prep.remove_invalid(depth)
        depth = self.prep.fill_holes(depth)
        depth = self.prep.bilateral_filter(depth)

        # Segment
        if self.inference is not None:
            seg = self.inference.hybrid_inference(depth)
        else:
            geo = self.geo_seg.isolate_foot(depth)
            seg = {
                "valid": geo["valid"],
                "mask": geo["mask"],
                "depth_isolated": geo["depth_isolated"],
                "conf": 1.0 if geo["valid"] else 0.0,
                "method": "geometric",
            }
        return seg

    # ------------------------------------------------------------------ #
    #  Full pipeline
    # ------------------------------------------------------------------ #

    def run(self, job_id: str, scan_dir: str, stl_out_path: str) -> dict:
        """
        Run full depth-only foot scan pipeline.

        Args:
            job_id:        unique job identifier
            scan_dir:      directory containing depth frames
            stl_out_path:  output STL file path

        Returns:
            dict matching v2 response schema
        """
        t_start = time.time()
        result = {
            "job_id": job_id,
            "status": "processing",
            "stages": {},
        }

        # ── Stage 1: Load frames ─────────────────────────────────────── #
        t0 = time.time()
        frames = self.load_depth_frames(scan_dir)
        if not frames:
            raise ValueError(f"No depth frames found in {scan_dir}")

        # FIX: Read actual camera intrinsics from the uploaded zip.
        # iOS stores calibration in camera_intrinsics.json (png16 mode)
        # OR in meta.json (bin mode, after v7.8+ fix).
        # Pipeline was using hardcoded defaults (cx=256, cy=192 = wrong for
        # 640×480) — this caused skewed/distorted 3D reconstruction.
        intr_path = Path(scan_dir) / "camera_intrinsics.json"
        meta_path = Path(scan_dir) / "meta.json"
        if intr_path.exists():
            try:
                with open(intr_path) as f:
                    intr = json.load(f)
                self.prep.fx = float(intr.get("fx", self.prep.fx))
                self.prep.fy = float(intr.get("fy", self.prep.fy))
                self.prep.cx = float(intr.get("cx", self.prep.cx))
                self.prep.cy = float(intr.get("cy", self.prep.cy))
                logger.info(f"✓ Intrinsics from camera_intrinsics.json: "
                            f"fx={self.prep.fx:.1f} fy={self.prep.fy:.1f} "
                            f"cx={self.prep.cx:.1f} cy={self.prep.cy:.1f}")
            except Exception as e:
                logger.warning(f"⚠ Could not read camera_intrinsics.json: {e}")
        elif meta_path.exists():
            # Bin mode: intrinsics stored in meta.json (v7.8+)
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "fx" in meta:
                    self.prep.fx = float(meta["fx"])
                    self.prep.fy = float(meta.get("fy", meta["fx"]))
                    self.prep.cx = float(meta.get("cx", self.prep.cx))
                    self.prep.cy = float(meta.get("cy", self.prep.cy))
                    logger.info(f"✓ Intrinsics from meta.json: "
                                f"fx={self.prep.fx:.1f} fy={self.prep.fy:.1f} "
                                f"cx={self.prep.cx:.1f} cy={self.prep.cy:.1f}")
                else:
                    logger.info("ℹ meta.json has no intrinsics (pre-v7.8) — using defaults")
            except Exception as e:
                logger.warning(f"⚠ Could not read meta.json intrinsics: {e}")
        else:
            logger.info(f"ℹ No intrinsics file found — using defaults "
                        f"fx={self.prep.fx} fy={self.prep.fy} "
                        f"cx={self.prep.cx} cy={self.prep.cy}")

        result["stages"]["load"] = {"frames": len(frames), "time": round(time.time() - t0, 2)}

        # ── Stage 2-4: Per-frame segmentation ───────────────────────── #
        t0 = time.time()
        valid_frames = []
        valid_masks = []
        for i, depth in enumerate(frames):
            seg = self.process_single_frame(depth)
            if seg["valid"]:
                valid_frames.append(seg["depth_isolated"])
                valid_masks.append(seg["mask"])
                logger.info(f"Frame {i+1}/{len(frames)}: ✓ ({seg['method']}, conf={seg['conf']:.2f})")
            else:
                logger.info(f"Frame {i+1}/{len(frames)}: ✗ no foot found")

        if not valid_frames:
            raise ValueError("No frames contained a valid foot")

        result["stages"]["segmentation"] = {
            "valid_frames": len(valid_frames),
            "total_frames": len(frames),
            "time": round(time.time() - t0, 2),
        }

        # ── Stage 5: 3D reconstruction ──────────────────────────────── #
        t0 = time.time()
        if len(valid_frames) >= 3:
            # Multi-frame fusion
            logger.info(f"Multi-frame fusion of {len(valid_frames)} frames")
            mesh = self.recon.fuse_frames(
                valid_frames,
                voxel_size=0.001,
                output_path=stl_out_path.replace(".stl", "_pre.obj"),
                method="poisson",
            )
        else:
            # Single-frame reconstruction
            logger.info("Single-frame reconstruction")
            mesh = self.recon.reconstruct_from_depth(
                valid_frames[0],
                output_path=stl_out_path.replace(".stl", "_pre.obj"),
                method="poisson",
            )
        result["stages"]["reconstruction"] = {"time": round(time.time() - t0, 2)}

        # ── Stage 6: Export STL ─────────────────────────────────────── #
        t0 = time.time()
        Path(stl_out_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(stl_out_path, mesh)
        result["stages"]["export"] = {"time": round(time.time() - t0, 2)}

        # ── Stage 7: Measurements ──────────────────────────────────── #
        measurements = self.recon.measure_foot(mesh)

        result["status"] = "completed"
        result["total_time"] = round(time.time() - t_start, 2)
        result["foot_length_mm"] = measurements["length_mm"]
        result["foot_width_mm"] = measurements["width_mm"]
        result["foot_height_mm"] = measurements["height_mm"]
        result["eu_size_approx"] = measurements["eu_size_approx"]
        result["mesh_vertices"] = measurements["vertices"]
        result["mesh_triangles"] = measurements["triangles"]
        result["stl_url"] = f"/v2/download-stl/{job_id}"
        result["method"] = "depth_only_hybrid" if self.inference else "depth_only_geometric"
        result["confidence_score"] = float(np.mean([
            len(valid_frames) / max(len(frames), 1),
            min(measurements["vertices"] / 5000.0, 1.0),
        ]))

        return result


# ======================================================================
#  SINGLETON
# ======================================================================

_pipeline_instance: Optional[DepthFootPipeline] = None


def get_depth_pipeline() -> DepthFootPipeline:
    """Lazy singleton — avoids loading YOLO weights on import."""
    global _pipeline_instance
    if _pipeline_instance is None:
        from ..config import settings  # uses your existing config
        _pipeline_instance = DepthFootPipeline(
            weights_dir=getattr(settings, "WEIGHTS_DIR", "weights"),
            fx=getattr(settings, "CAMERA_FX", 585.0),
            fy=getattr(settings, "CAMERA_FY", 585.0),
            cx=getattr(settings, "CAMERA_CX", 256.0),
            cy=getattr(settings, "CAMERA_CY", 192.0),
        )
    return _pipeline_instance
