"""
backend/app/services/depth_pipeline.py

Refactored Depth-only foot scan pipeline orchestrator.
Optimized for 2D temporal fusion and exact-contour mesh output pipelines.
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


class DepthFootPipeline:
    """
    Depth-only foot scanning execution manager pipeline.
    """

    def __init__(
        self,
        weights_dir: str = "weights",
        yolo_model_name: str = "foot_yolov8_seg.pt",
        # TrueDepth depth maps are ~480x640 PORTRAIT with fx~432, cx~240,
        # cy~320. The previous defaults (585/320/240) were 640x480 RGB-camera
        # values — wrong axis AND wrong focal length, which scaled every
        # back-projected coordinate by ~1.35x and skewed the principal point.
        # That produced 600-820mm "feet" whenever meta.json lacked fx.
        fx: float = 432.0,
        fy: float = 432.0,
        cx: float = 240.0,
        cy: float = 320.0,
    ):
        self.prep = DepthPreprocessor(fx=fx, fy=fy, cx=cx, cy=cy)
        self.floor = FloorRemover(self.prep)
        self.geo_seg = FootDepthSegmenter(self.prep, self.floor)
        self.recon = FootReconstructor(self.prep)

        yolo_path = Path(weights_dir) / yolo_model_name
        self.inference: Optional[FootScanInference] = None
        if yolo_path.exists():
            try:
                self.inference = FootScanInference(str(yolo_path), self.prep, self.geo_seg)
                logger.info(f"✓ YOLO model loaded: {yolo_path}")
            except Exception as e:
                logger.warning(f"⚠ YOLO load failed ({e}) — using geometric mode fallback")
        else:
            logger.info("ℹ Running system under geometric-only processing metrics")

    def load_depth_frames(self, scan_dir: str) -> List[np.ndarray]:
        scan_dir = Path(scan_dir)
        frames = []
        patterns = ["depth_*.png", "*.depth.png", "frame_*.png", "*.tiff", "*.npy", "depth_*.bin"]
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
                    txt = f.with_suffix(".txt")
                    if txt.exists():
                        dims = txt.read_text().strip().split(",")
                        w_d, h_d = int(dims[0]), int(dims[1])
                        arr = np.frombuffer(f.read_bytes(), dtype=np.float32).reshape(h_d, w_d)
                        frames.append(arr)
                    else:
                        logger.warning(f"Missing text spatial bindings for bin data: {f.name}")
                else:
                    depth = self.prep.load_depth(str(f))
                    frames.append(depth)
            except Exception as e:
                logger.warning(f"Skipping corrupt resource entry {f.name}: {e}")

        logger.info(f"Loaded total of {len(frames)} operational depth frame sources.")
        return frames

    def process_single_frame(self, depth: np.ndarray) -> dict:
        # ORDER MATTERS. The previous order ran fill_holes BEFORE
        # segmentation — but fill_holes (TELEA inpaint) fills the ENTIRE
        # frame, not just holes inside the foot, turning all 307k pixels into
        # "valid depth". The nearest-depth-band segmenter then saw a full
        # frame and failed. Fix: segment on the RAW (invalid-removed) depth
        # first so the nearest-surface foot blob is found cleanly, THEN fill
        # holes and refine only WITHIN the foot mask.
        depth = self.prep.remove_invalid(depth)

        if self.inference is not None:
            # YOLO trained on SYNTHETIC depth only — unreliable on real data
            # (boxes spanning leg+foot+floor → inflated measurements). Use it
            # only as a VALIDATOR of the geometric mask, never to expand it.
            # Flip USE_YOLO_VALIDATOR on after retraining on real depth.
            USE_YOLO_VALIDATOR = False
            geo = self.geo_seg.isolate_foot(depth)
            seg = {
                "valid": geo["valid"],
                "mask": geo["mask"],
                "depth_isolated": geo["depth_isolated"],
                "conf": 1.0 if geo["valid"] else 0.0,
                "method": "geometric",
            }
            if USE_YOLO_VALIDATOR and geo["valid"]:
                seg = self._validate_with_yolo(depth, seg)
        else:
            geo = self.geo_seg.isolate_foot(depth)
            seg = {
                "valid": geo["valid"],
                "mask": geo["mask"],
                "depth_isolated": geo["depth_isolated"],
                "conf": 1.0 if geo["valid"] else 0.0,
                "method": "geometric",
            }

        if not seg.get("valid") or seg.get("mask") is None:
            return seg

        # Smooth silhouette, then fill holes + edge-refine ONLY inside foot.
        smooth_mask = self.prep.smooth_silhouette_mask(seg["mask"])
        di = seg["depth_isolated"].copy()
        di[smooth_mask < 128] = np.nan
        # Fill holes inside the foot region, refine, then re-clip to mask so
        # nothing leaks outside the silhouette.
        di = self.prep.fill_holes(di)
        di = self.prep.edge_aware_refine(di, passes=2)
        di[smooth_mask < 128] = np.nan
        seg["mask"] = smooth_mask
        seg["depth_isolated"] = di
        return seg

    def _validate_with_yolo(self, depth: np.ndarray, seg: dict) -> dict:
        """
        Use YOLO to confirm the geometric blob is foot-shaped — never to
        expand it. If YOLO sees a foot overlapping the geometric mask, raise
        confidence. If YOLO disagrees (box elsewhere / huge), keep geometric
        but flag low confidence. YOLO cannot add pixels the geometric mask
        didn't already select.
        """
        try:
            img_3ch = self.prep.to_pseudo_rgb(depth)
            y = self.inference.run_yolo(img_3ch)
        except Exception:
            return seg
        if y.get("mask") is None:
            seg["conf"] = 0.5
            return seg
        geo_mask = seg["mask"] > 0
        yolo_mask = y["mask"] > 0
        inter = (geo_mask & yolo_mask).sum()
        union = (geo_mask | yolo_mask).sum()
        iou = inter / union if union > 0 else 0.0
        seg["conf"] = float(max(0.3, min(1.0, iou + 0.3)))
        # YOLO only ever RESTRICTS to the agreed region, never expands:
        if iou > 0.3:
            agreed = (geo_mask & yolo_mask).astype('uint8') * 255
            if (agreed > 0).sum() > 1000:
                seg["mask"] = agreed
                di = seg["depth_isolated"].copy()
                di[agreed < 128] = np.nan
                seg["depth_isolated"] = di
        return seg

    @staticmethod
    def _write_ascii_stl(mesh, path: str):
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            raise ValueError("Mismatched data fields: triangle parameters cannot evaluate null matrices.")

        with open(path, "w") as f:
            f.write("solid digifoot\n")
            for t in tris:
                v0, v1, v2 = verts[t[0]], verts[t[1]], verts[t[2]]
                n = np.cross(v1 - v0, v2 - v0)
                ln = np.linalg.norm(n)
                n = n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in (v0, v1, v2):
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid digifoot\n")

    def run(self, job_id: str, scan_dir: str, stl_out_path: str) -> dict:
        t_start = time.time()
        result = {
            "job_id": job_id,
            "status": "processing",
            "stages": {},
        }

        # Load Frames & Intrinsics
        t0 = time.time()
        frames = self.load_depth_frames(scan_dir)
        if not frames:
            raise ValueError(f"No usable depth array information within runtime directory: {scan_dir}")

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
            except Exception as e:
                logger.warning(f"Error parse intrinsics config data: {e}")
        elif meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "fx" in meta:
                    self.prep.fx = float(meta["fx"])
                    self.prep.fy = float(meta.get("fy", meta["fx"]))
                    self.prep.cx = float(meta.get("cx", self.prep.cx))
                    self.prep.cy = float(meta.get("cy", self.prep.cy))
            except Exception as e:
                logger.warning(f"Error parse layout metadata parameters: {e}")

        result["stages"]["load"] = {"frames": len(frames), "time": round(time.time() - t0, 2)}

        h_d, w_d = frames[0].shape[:2]
        center_cx, center_cy = w_d / 2.0, h_d / 2.0
        # The principal point of a depth map sits at (or within a few px of)
        # the image center. If the configured cx/cy are off-center by more
        # than ~12% of the dimension, or fx is implausible for this width,
        # the intrinsics are wrong/stale (e.g. 585/320/240 RGB defaults on a
        # 480x640 depth map) → recompute from FOV. This MUST catch the
        # 585/320/240 case that produced 600-820mm feet.
        import math
        bad_cx = abs(self.prep.cx - center_cx) > 0.12 * w_d
        bad_cy = abs(self.prep.cy - center_cy) > 0.12 * h_d
        # Plausible TrueDepth fx for a 480-wide depth map is ~400-470.
        # Express as a ratio to width: fx/width ≈ 0.85-1.0. Flag if outside
        # 0.7-1.25 of the short axis.
        short = float(min(w_d, h_d))
        f_ratio = self.prep.fx / short
        bad_f = not (0.70 <= f_ratio <= 1.25)
        if bad_cx or bad_cy or bad_f:
            est_f = short / (2.0 * math.tan(math.radians(55.0) / 2.0))
            old = (self.prep.fx, self.prep.cx, self.prep.cy)
            self.prep.fx = est_f
            self.prep.fy = est_f
            self.prep.cx = center_cx
            self.prep.cy = center_cy
            logger.warning(
                f"Intrinsics {old} implausible for {w_d}x{h_d} "
                f"(bad_cx={bad_cx} bad_cy={bad_cy} bad_f={bad_f}) → "
                f"recomputed fx={self.prep.fx:.1f} cx={self.prep.cx:.1f} cy={self.prep.cy:.1f}")
        logger.info(f"Using intrinsics fx={self.prep.fx:.1f} fy={self.prep.fy:.1f} "
                    f"cx={self.prep.cx:.1f} cy={self.prep.cy:.1f} on {w_d}x{h_d} frames")

        # Downsample Frame Rates uniformly to handle CPU bottlenecks cleanly
        MAX_FRAMES = 18
        if len(frames) > MAX_FRAMES:
            idx = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
            frames = [frames[i] for i in idx]

        # Segmentation Pass
        t0 = time.time()
        valid_frames = []
        valid_masks = []
        for i, depth in enumerate(frames):
            try:
                seg = self.process_single_frame(depth)
                if seg["valid"]:
                    valid_frames.append(seg["depth_isolated"])
                    valid_masks.append(seg["mask"])
            except Exception as e:
                logger.warning(f"Fault condition parsed at frame context index {i}: {e}")
                continue

        if not valid_frames:
            raise ValueError("System isolated zero valid structural contours across execution stack bounds.")

        result["stages"]["segmentation"] = {
            "valid_frames": len(valid_frames),
            "total_frames": len(frames),
            "time": round(time.time() - t0, 2),
        }

        # 3D Mesh Generation & Formatting
        t0 = time.time()

        # v8.1 FIX: NO upsampling. The previous version doubled the depth
        # map resolution with joint_bilateral_upsample, but self.prep.fx/cx/cy
        # stayed at the ORIGINAL camera intrinsics — so back-projection
        # stretched world coordinates by ~2× (and worse at frame edges),
        # producing meshes with 700-1900mm "foot lengths". Bug seen in jobs
        # 667939d6, eb38dc20, e644d603, f89b0364 (all 2026-05-30).
        # Fix: stop upsampling. The Poisson + edge-aware refine already give
        # smooth output; the upsample-for-anti-staircase was speculative
        # gain at the cost of unit-correctness.
        fused_mask = valid_masks[0].copy()
        for m in valid_masks[1:]:
            fused_mask = cv2.bitwise_or(fused_mask, m)

        if len(valid_frames) >= 3:
            logger.info(f"Stationary fusion over {len(valid_frames)} frames.")
            mesh = self.recon.fuse_stationary_frames(
                valid_frames,
                output_path=stl_out_path.replace(".stl", "_pre.obj"),
                target_triangles=60_000,
                mask=fused_mask,
            )
        else:
            logger.info("Reconstructing single frame.")
            mesh = self.recon.reconstruct_from_depth(
                valid_frames[0],
                output_path=stl_out_path.replace(".stl", "_pre.obj"),
                target_triangles=60_000,
                mask=valid_masks[0],
            )
        result["stages"]["reconstruction"] = {"time": round(time.time() - t0, 2)}

        # Exposing Normals to Ensure Valid CAD Output Format Writes
        t0 = time.time()
        Path(stl_out_path).parent.mkdir(parents=True, exist_ok=True)

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()  

        ok = o3d.io.write_triangle_mesh(stl_out_path, mesh, write_triangle_uvs=False)
        stl_file = Path(stl_out_path)
        if not ok or not stl_file.exists() or stl_file.stat().st_size == 0:
            logger.warning("File streams unallocated via backend drivers. Compiling raw ASCII backups manually.")
            self._write_ascii_stl(mesh, stl_out_path)

        final_size = Path(stl_out_path).stat().st_size if Path(stl_out_path).exists() else 0
        result["stages"]["export"] = {
            "time": round(time.time() - t0, 2),
            "stl_bytes": final_size,
        }

        # Run Extraction Tasks against standardized coordinate arrays
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


# Singleton Initialization
_pipeline_instance: Optional[DepthFootPipeline] = None

def get_depth_pipeline() -> DepthFootPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        from ..config import settings  
        _pipeline_instance = DepthFootPipeline(
            weights_dir=getattr(settings, "WEIGHTS_DIR", "weights"),
            fx=getattr(settings, "CAMERA_FX", 585.0),
            fy=getattr(settings, "CAMERA_FY", 585.0),
            cx=getattr(settings, "CAMERA_CX", 320.0),  
            cy=getattr(settings, "CAMERA_CY", 240.0),  
        )
    return _pipeline_instance