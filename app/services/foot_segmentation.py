# foot_segmentation.py
# Geometric foot isolation from depth maps
# Hybrid: rule-based geometric + YOLOv8-seg refinement

import numpy as np
import cv2
import open3d as o3d
from typing import Optional, List, Tuple
from ultralytics import YOLO


# ======================================================================
#  GEOMETRIC SEGMENTER
# ======================================================================

class FootDepthSegmenter:
    """
    Rule-based geometric foot isolation.
    Works with zero training data. Use as fallback or ROI generator.
    """

    def __init__(self, preprocessor, floor_remover):
        self.prep = preprocessor
        self.floor = floor_remover

    def depth_threshold_mask(
        self,
        depth: np.ndarray,
        z_min: float = 0.25,
        z_max: float = 1.20,
    ) -> np.ndarray:
        """Binary mask for valid depth range."""
        mask = (~np.isnan(depth)) & (depth >= z_min) & (depth <= z_max)
        return mask.astype(np.uint8) * 255

    def morphological_clean(
        self,
        mask: np.ndarray,
        open_k: int = 5,
        close_k: int = 15,
    ) -> np.ndarray:
        """Opening removes noise, closing fills holes in binary mask."""
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        return mask

    def largest_blob(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component (most likely the foot)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return ((labels == largest) * 255).astype(np.uint8)

    def aspect_ratio_filter(
        self,
        mask: np.ndarray,
        min_ratio: float = 1.5,
        max_ratio: float = 4.5,
    ) -> Optional[np.ndarray]:
        """
        Foot aspect ratio ≈ 2–4:1 (length:width).
        Rejects non-foot objects (round blobs = hand, near-square = leg cross-section).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        w, h_r = rect[1]
        if w == 0 or h_r == 0:
            return None

        ratio = max(w, h_r) / min(w, h_r)
        if min_ratio <= ratio <= max_ratio:
            return mask
        return None

    def size_filter(
        self,
        mask: np.ndarray,
        min_area_frac: float = 0.02,
        max_area_frac: float = 0.40,
    ) -> Optional[np.ndarray]:
        """Reject objects too small or too large to be a foot."""
        total = mask.shape[0] * mask.shape[1]
        foot_area = (mask > 0).sum()
        frac = foot_area / total
        if min_area_frac <= frac <= max_area_frac:
            return mask
        return None

    def roughness_filter(
        self,
        depth_isolated: np.ndarray,
        min_roughness: float = 0.002,
    ) -> bool:
        """
        Socks/slippers have smoother depth surface than bare feet.
        Returns True if surface is rough enough (bare foot likely).
        """
        valid = depth_isolated[~np.isnan(depth_isolated)]
        if len(valid) < 100:
            return False
        return float(np.std(valid)) >= min_roughness

    def extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract foot boundary contours, sorted by area (largest first)."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        return sorted(contours, key=cv2.contourArea, reverse=True)

    def isolate_foot(self, depth: np.ndarray) -> dict:
        """
        Full geometric foot isolation pipeline.
        Returns dict: depth_isolated, mask, contours, valid, depth_no_floor
        """
        # Remove floor + out-of-range depth
        depth_nf = self.floor.remove_floor_from_depth(depth)
        depth_nf = self.floor.height_threshold_removal(depth_nf)

        # Threshold mask + morphological cleanup
        mask = self.depth_threshold_mask(depth_nf)
        mask = self.morphological_clean(mask)
        mask = self.largest_blob(mask)

        # Validate shape + size
        mask = self.size_filter(mask) or mask
        valid_mask = self.aspect_ratio_filter(mask)

        # Apply mask to depth
        depth_isolated = depth.copy()
        if valid_mask is not None:
            depth_isolated[valid_mask == 0] = np.nan
            contours = self.extract_contours(valid_mask)
        else:
            depth_isolated[:] = np.nan
            contours = []

        return {
            "depth_isolated": depth_isolated,
            "mask": valid_mask if valid_mask is not None else np.zeros_like(mask),
            "contours": contours,
            "depth_no_floor": depth_nf,
            "valid": valid_mask is not None,
        }


# ======================================================================
#  ADVANCED FLOOR / BACKGROUND REMOVAL
# ======================================================================

class AdvancedFloorRemover:
    """Additional floor/background removal strategies."""

    def remove_by_normal(
        self,
        pcd: o3d.geometry.PointCloud,
        floor_normal_threshold: float = 0.9,
    ) -> o3d.geometry.PointCloud:
        """
        Estimate surface normals, remove floor points.
        Floor normals ≈ (0, ±1, 0) — Y-dominant.
        """
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=30
            )
        )
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        non_floor = np.abs(normals[:, 1]) < floor_normal_threshold

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points[non_floor])
        if pcd.has_normals():
            filtered.normals = o3d.utility.Vector3dVector(normals[non_floor])
        return filtered

    def remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> o3d.geometry.PointCloud:
        """Remove flying pixels and statistical outliers."""
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return pcd.select_by_index(ind)

    def remove_radius_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_points: int = 16,
        radius: float = 0.05,
    ) -> o3d.geometry.PointCloud:
        """Remove isolated points with few neighbors in radius."""
        _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return pcd.select_by_index(ind)

    def depth_discontinuity_edges(
        self,
        depth: np.ndarray,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Detect object/background boundaries via depth gradient.
        Large gradient → depth discontinuity → edge pixel.
        """
        dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(dx ** 2 + dy ** 2)
        return (edge_mag > threshold).astype(np.uint8) * 255


# ======================================================================
#  HYBRID INFERENCE
# ======================================================================

class FootScanInference:
    """
    Hybrid inference: geometric coarse segmentation → YOLOv8-seg refinement.
    Falls back to geometric if YOLO confidence is low.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor,
        geo_segmenter: FootDepthSegmenter,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
    ):
        self.yolo = YOLO(model_path)
        self.prep = preprocessor
        self.geo = geo_segmenter
        self.conf_thresh = conf_threshold
        self.iou_thresh = iou_threshold

    def run_yolo(self, img_3ch: np.ndarray) -> dict:
        """
        Run YOLOv8-seg inference on 3-channel depth pseudo-RGB image.
        Returns best detection mask + confidence.
        """
        results = self.yolo(
            img_3ch,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )

        if not results or results[0].masks is None:
            return {"mask": None, "conf": 0.0, "bbox": None}

        r = results[0]
        best_idx = int(r.boxes.conf.argmax())
        mask_data = r.masks.data[best_idx].cpu().numpy()
        mask_uint8 = (mask_data * 255).astype(np.uint8)
        h, w = img_3ch.shape[:2]
        mask_resized = cv2.resize(
            mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST
        )
        conf = float(r.boxes.conf[best_idx])
        bbox = r.boxes.xyxy[best_idx].cpu().numpy()

        return {"mask": mask_resized, "conf": conf, "bbox": bbox}

    def hybrid_inference(self, depth: np.ndarray) -> dict:
        """
        Two-stage pipeline:
          Stage 1 — Geometric isolation (fast, always runs)
          Stage 2 — YOLOv8-seg refinement (runs on full frame)
          Combine: AND of both masks for maximum precision.
        """
        # Stage 1: geometric
        geo_result = self.geo.isolate_foot(depth)

        # Stage 2: YOLO
        img_3ch = self.prep.to_pseudo_rgb(depth)
        yolo_result = self.run_yolo(img_3ch)

        # Select best mask
        if yolo_result["mask"] is not None and yolo_result["conf"] > self.conf_thresh:
            final_mask = yolo_result["mask"]
            method = "yolo"
        elif geo_result["valid"]:
            final_mask = geo_result["mask"]
            method = "geometric"
        else:
            return {"valid": False, "method": "none"}

        # Combine if both available (highest precision)
        if (
            yolo_result["mask"] is not None
            and geo_result["valid"]
            and geo_result["mask"] is not None
        ):
            combined = cv2.bitwise_and(yolo_result["mask"], geo_result["mask"])
            if combined.sum() > 1000:
                final_mask = combined
                method = "hybrid"

        # Apply mask to depth
        depth_isolated = depth.copy()
        depth_isolated[final_mask < 128] = np.nan

        return {
            "valid": True,
            "mask": final_mask,
            "depth_isolated": depth_isolated,
            "conf": yolo_result["conf"],
            "method": method,
            "geo_result": geo_result,
        }
