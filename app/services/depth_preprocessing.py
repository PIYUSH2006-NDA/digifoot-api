# depth_preprocessing.py
# Depth-only foot scan preprocessing pipeline
# Compatible with iPhone TrueDepth + LiDAR sensors

import numpy as np
import cv2
import open3d as o3d
from scipy import ndimage
from typing import Tuple, Optional, Union


class DepthPreprocessor:
    """
    Core depth map preprocessor for TrueDepth / LiDAR input.
    All distances in meters.
    """

    def __init__(
        self,
        fx: float = 585.0,   # TrueDepth focal length X (pixels)
        fy: float = 585.0,   # TrueDepth focal length Y (pixels)
        cx: float = 320.0,   # FIX: was 256.0 — correct principal point for 640×480
        cy: float = 240.0,   # FIX: was 192.0 — correct principal point for 640×480
        depth_scale: float = 1000.0,  # uint16 raw → meters divisor
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load_depth(self, path: str) -> np.ndarray:
        """Load 16-bit PNG depth map from disk (TrueDepth output)."""
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Cannot load depth image: {path}")
        return depth.astype(np.float32) / self.depth_scale  # → meters

    def load_depth_from_array(self, raw: np.ndarray) -> np.ndarray:
        """Accept raw uint16 numpy array (from AVDepthData on iOS)."""
        return raw.astype(np.float32) / self.depth_scale

    # ------------------------------------------------------------------ #
    #  Cleaning
    # ------------------------------------------------------------------ #

    def remove_invalid(self, depth: np.ndarray,
                        max_range: float = 5.0) -> np.ndarray:
        """Set invalid pixels (zero / out-of-range) to NaN."""
        depth = depth.copy()
        depth[depth <= 0] = np.nan
        depth[depth > max_range] = np.nan
        return depth

    def fill_holes(self, depth: np.ndarray) -> np.ndarray:
        """Inpaint NaN holes using TELEA fast marching method."""
        mask_nan = np.isnan(depth).astype(np.uint8)
        depth_filled = np.nan_to_num(depth, nan=0.0)
        # Scale to uint8 for inpaint (OpenCV limitation)
        max_val = np.nanmax(depth_filled) if np.nanmax(depth_filled) > 0 else 1.0
        depth_uint8 = cv2.normalize(
            depth_filled, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        inpainted = cv2.inpaint(depth_uint8, mask_nan, 5, cv2.INPAINT_TELEA)
        # Rescale back to meters
        result = inpainted.astype(np.float32) / 255.0 * max_val
        result = np.where(mask_nan, result, depth_filled)
        return result

    # ------------------------------------------------------------------ #
    #  Filtering
    # ------------------------------------------------------------------ #

    def bilateral_filter(
        self,
        depth: np.ndarray,
        d: int = 5,
        sigma_color: float = 0.1,
        sigma_space: float = 5.0,
    ) -> np.ndarray:
        """
        Edge-preserving bilateral filter.
        Smooths depth noise while preserving object boundaries.
        """
        valid_mask = ~np.isnan(depth)
        depth_clean = np.nan_to_num(depth, nan=0.0)
        filtered = cv2.bilateralFilter(
            depth_clean.astype(np.float32), d, sigma_color, sigma_space
        )
        filtered[~valid_mask] = np.nan
        return filtered

    def gaussian_smooth(self, depth: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Light Gaussian smoothing for high-frequency sensor noise."""
        valid = ~np.isnan(depth)
        d = np.nan_to_num(depth, nan=0.0)
        smoothed = cv2.GaussianBlur(d, (ksize, ksize), 0)
        smoothed[~valid] = np.nan
        return smoothed

    def median_filter(self, depth: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Median filter — good for salt-and-pepper noise."""
        valid = ~np.isnan(depth)
        d = np.nan_to_num(depth, nan=0.0).astype(np.float32)
        filtered = cv2.medianBlur(d, ksize)
        filtered[~valid] = np.nan
        return filtered

    # ------------------------------------------------------------------ #
    #  Normalization
    # ------------------------------------------------------------------ #

    def normalize_depth(
        self,
        depth: np.ndarray,
        min_d: float = 0.2,
        max_d: float = 1.5,
    ) -> np.ndarray:
        """Clip depth to scan range and normalize to [0, 1]."""
        clipped = np.clip(depth, min_d, max_d)
        norm = (clipped - min_d) / (max_d - min_d)
        norm = np.nan_to_num(norm, nan=0.0)
        return norm.astype(np.float32)

    def to_8bit(self, depth_norm: np.ndarray) -> np.ndarray:
        """Convert [0,1] normalized depth to uint8 for visualization / YOLO."""
        return (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)

    def to_pseudo_rgb(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth to 3-channel pseudo-RGB for YOLO input:
          Ch0 = normalized depth
          Ch1 = surface normal magnitude (Sobel)
          Ch2 = local curvature (Laplacian)
        """
        norm = self.normalize_depth(depth)
        ch0 = self.to_8bit(norm)

        gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        normal_mag = np.sqrt(gx ** 2 + gy ** 2)
        ch1 = cv2.normalize(normal_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        lap = cv2.Laplacian(norm, cv2.CV_32F, ksize=3)
        ch2 = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return cv2.merge([ch0, ch1, ch2])

    # ------------------------------------------------------------------ #
    #  Full pipeline
    # ------------------------------------------------------------------ #

    def preprocess_full(
        self,
        path_or_array: Union[str, np.ndarray],
        return_pseudo_rgb: bool = True,
    ) -> dict:
        """
        End-to-end preprocessing:
          load → remove invalid → fill holes → bilateral filter → normalize

        Returns dict with keys:
          raw_meters, normalized, uint8, pseudo_rgb (optional)
        """
        if isinstance(path_or_array, str):
            depth = self.load_depth(path_or_array)
        else:
            depth = self.load_depth_from_array(path_or_array)

        depth = self.remove_invalid(depth)
        depth = self.fill_holes(depth)
        depth = self.bilateral_filter(depth)
        depth_norm = self.normalize_depth(depth)

        result = {
            "raw_meters": depth,
            "normalized": depth_norm,
            "uint8": self.to_8bit(depth_norm),
        }
        if return_pseudo_rgb:
            result["pseudo_rgb"] = self.to_pseudo_rgb(depth)

        return result


class FloorRemover:
    """RANSAC plane detection + floor removal from depth maps."""

    def __init__(self, preprocessor: DepthPreprocessor):
        self.prep = preprocessor

    def depth_to_pointcloud(self, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """Project depth map → Open3D PointCloud using pinhole model."""
        h, w = depth.shape
        i_coords, j_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        valid = ~np.isnan(depth) & (depth > 0)
        z = depth[valid]
        x = (j_coords[valid] - self.prep.cx) * z / self.prep.fx
        y = (i_coords[valid] - self.prep.cy) * z / self.prep.fy

        pts = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def ransac_floor_plane(
        self,
        pcd: o3d.geometry.PointCloud,
        dist_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC plane segmentation. Returns (plane_model [a,b,c,d], inlier_indices)."""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        return np.array(plane_model), np.array(inliers)

    def remove_floor_from_depth(
        self,
        depth: np.ndarray,
        margin_above: float = 0.03,
    ) -> np.ndarray:
        """
        Remove floor pixels using RANSAC plane detection.
        margin_above: keep points this many meters above floor plane.
        """
        pcd = self.depth_to_pointcloud(depth)

        if len(pcd.points) < 100:
            return depth

        plane_model, _ = self.ransac_floor_plane(pcd)
        a, b, c, d = plane_model
        norm_len = np.sqrt(a ** 2 + b ** 2 + c ** 2)

        h, w = depth.shape
        i_coords, j_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        valid = ~np.isnan(depth) & (depth > 0)
        z = np.nan_to_num(depth, nan=0.0)
        x = (j_coords - self.prep.cx) * z / self.prep.fx
        y = (i_coords - self.prep.cy) * z / self.prep.fy

        # Signed distance from plane
        plane_dist = (a * x + b * y + c * z + d) / norm_len

        # Floor: within margin of plane (or behind it)
        floor_mask = (plane_dist < margin_above) & valid

        depth_no_floor = depth.copy()
        depth_no_floor[floor_mask] = np.nan
        return depth_no_floor

    def height_threshold_removal(
        self,
        depth: np.ndarray,
        min_depth: float = 0.20,
        max_depth: float = 1.50,
    ) -> np.ndarray:
        """Simple depth-range clip — removes far background and very near objects."""
        result = depth.copy()
        result[depth < min_depth] = np.nan
        result[depth > max_depth] = np.nan
        return result

    def background_subtraction(
        self,
        depth: np.ndarray,
        background: np.ndarray,
        diff_threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Foreground mask via background subtraction.
        Requires a calibration frame (empty scene without foot).
        diff > threshold → foreground pixel.
        """
        diff = background - depth  # positive = closer to camera
        fg_mask = (diff > diff_threshold).astype(np.uint8) * 255
        return fg_mask