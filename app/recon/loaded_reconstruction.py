"""
backend_loaded_reconstruction.py

Multi-view 3D reconstruction for LOADED-STATE foot scans.

Input  : the zip produced by LoadedScanEngine (iOS), containing
             manifest.json
             angleNN/depth_KK.png      16-bit depth PNG, value = mm
             angleNN/meta.json         intrinsics, attitude quaternion, label
Output : a watertight foot mesh (.ply) + a registration report.

This module targets ~2 mm reconstruction accuracy. The accuracy budget:
    * TrueDepth per-pixel noise   ~0.5-1.5 mm at 25-40 cm  -> reduced by
      burst temporal averaging (sqrt(N) gain; 6 frames -> ~2.4x)
    * pairwise registration       point-to-plane ICP, sub-mm with good overlap
    * global consistency          pose-graph optimisation over all angles
    * surface integration         TSDF voxel 1 mm  OR  Poisson depth 10

CRITICAL ACCURACY NOTE
----------------------
2 mm rigid multi-view reconstruction REQUIRES the foot to be rigid between
captured angles. If the foot rotates under body weight between shots it
deforms and rigid ICP will not reach 2 mm. For 2 mm, capture with the foot
PLANTED and the phone moving around it. The attitude quaternion stored per
angle is the registration seed for that case. If you keep the phone static
and rotate the foot, expect ~4-6 mm and consider a non-rigid refinement
stage (not implemented here).

Dependencies: open3d>=0.18, numpy, pillow, scipy
"""

from __future__ import annotations

import json
import zipfile
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AngleCapture:
    index: int
    label: str
    intrinsics: dict          # fx, fy, cx, cy
    width: int
    height: int
    attitude_quat: list       # [x, y, z, w]
    depth_frames: list = field(default_factory=list)  # list[np.ndarray] meters
    mask_frames: list = field(default_factory=list)   # list[np.ndarray] bool
    mask_w: int = 0
    mask_h: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Zip loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scan_zip(zip_path: str) -> list[AngleCapture]:
    """Parse the LoadedScanEngine zip into per-angle captures."""
    angles: dict[int, AngleCapture] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Per-angle meta + depth bursts.
        for name in names:
            parts = name.split("/")
            if len(parts) != 2 or not parts[0].startswith("angle"):
                continue
            idx = int(parts[0][5:])

            if parts[1] == "meta.json":
                meta = json.loads(zf.read(name))
                angles.setdefault(idx, AngleCapture(
                    index=idx,
                    label=meta.get("label", ""),
                    intrinsics=meta["intrinsics"],
                    width=meta["depth_width"],
                    height=meta["depth_height"],
                    attitude_quat=meta.get("attitude_quat", [0, 0, 0, 1]),
                    mask_w=meta.get("mask_width", 0),
                    mask_h=meta.get("mask_height", 0),
                ))

        # Second pass: depth frames + stored masks (meta must exist first).
        for name in sorted(names):
            parts = name.split("/")
            if len(parts) != 2:
                continue
            idx = int(parts[0][5:]) if parts[0].startswith("angle") else -1
            if idx not in angles:
                continue

            if parts[1].endswith(".png"):
                depth_m = _decode_depth_png(zf.read(name))
                angles[idx].depth_frames.append(depth_m)

            elif parts[1].startswith("mask_") and parts[1].endswith(".bin"):
                a = angles[idx]
                if a.mask_w > 0 and a.mask_h > 0:
                    raw = np.frombuffer(zf.read(name), dtype=np.uint8)
                    if raw.size == a.mask_w * a.mask_h:
                        a.mask_frames.append(
                            raw.reshape(a.mask_h, a.mask_w) > 127)

    return [angles[i] for i in sorted(angles)]


def _decode_depth_png(png_bytes: bytes) -> np.ndarray:
    """Decode a 16-bit depth PNG (value = millimeters) into meters."""
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.asarray(img).astype(np.float32)   # uint16 -> float
    return arr / 1000.0                        # mm -> m


# ─────────────────────────────────────────────────────────────────────────────
# 2. Burst fusion — temporal noise reduction
# ─────────────────────────────────────────────────────────────────────────────

def fuse_burst(depth_frames: list[np.ndarray]) -> np.ndarray:
    """
    Combine a burst of depth frames of the same static view into one clean
    depth map. Per-pixel median is outlier-robust; averaging the inliers
    around the median gains ~sqrt(N) SNR.
    """
    stack = np.stack(depth_frames, axis=0)        # (N, H, W)
    valid = (stack > 0.05) & (stack < 1.5)
    stack_nan = np.where(valid, stack, np.nan)

    med = np.nanmedian(stack_nan, axis=0)         # robust centre
    # Keep samples within 5 mm of the median, then average them.
    inlier = np.abs(stack_nan - med[None]) < 0.005
    stack_in = np.where(inlier, stack_nan, np.nan)
    fused = np.nanmean(stack_in, axis=0)
    return np.nan_to_num(fused, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Foot segmentation
# ─────────────────────────────────────────────────────────────────────────────
#
# Mask priority per angle:
#   1. stored masks from the iOS zip  (YOLOv8-seg or geometric, already on
#      device) — preferred, no recompute, consistent with what the user saw
#   2. backend YOLOv8-seg              — `_yolo_foot_mask`, sharper refinement
#   3. geometric fallback             — `segment_foot`, no model needed
#
# Person 2's backend YOLOv8-seg integration plugs in at `_yolo_foot_mask`.

# Canonical depth normalisation — MUST match FootSegmentationRefiner.swift
# and the training-data renderer exactly.
_DEPTH_MIN_M = 0.15
_DEPTH_MAX_M = 0.85


def _depth_to_model_input(depth_m: np.ndarray) -> np.ndarray:
    """
    Depth (meters) -> 3-channel uint8 image, the canonical YOLO input.

      1. clamp to [0.15, 0.85] m
      2. normalise to 0..1 ; invalid/zero depth -> 0
      3. scale to 0..255 uint8
      4. replicate to 3 channels (H, W, 3)

    Identical to FootSegmentationRefiner.depthToInputPixelBuffer (minus the
    letterbox — ultralytics letterboxes internally on predict()).
    """
    rng = _DEPTH_MAX_M - _DEPTH_MIN_M
    v = (depth_m - _DEPTH_MIN_M) / rng
    valid = (depth_m >= _DEPTH_MIN_M) & (depth_m <= _DEPTH_MAX_M)
    v = np.where(valid, np.clip(v, 0.0, 1.0), 0.0)
    u8 = (v * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def fuse_masks(mask_frames: list[np.ndarray]) -> np.ndarray | None:
    """Majority-vote a burst of boolean masks into one stable mask."""
    if not mask_frames:
        return None
    stack = np.stack(mask_frames, axis=0).astype(np.float32)
    return stack.mean(axis=0) >= 0.5


def segment_foot(depth_m: np.ndarray, yolo_model=None) -> np.ndarray:
    """
    Backend foot mask when no on-device mask is available.

    Uses YOLOv8-seg if a model is supplied, else a geometric fallback (the
    same principle as FootDetectorV7: depth-histogram peak + ground removal).
    """
    if yolo_model is not None:
        m = _yolo_foot_mask(depth_m, yolo_model)
        if m is not None and m.any():
            return m
        # fall through to geometric if YOLO finds nothing

    valid = (depth_m > 0.05) & (depth_m < 1.5)
    d = depth_m[valid]
    if d.size < 500:
        return np.zeros_like(depth_m, dtype=bool)

    # Histogram peak = dominant surface (the foot, held closest/centred).
    hist, edges = np.histogram(d, bins=64)
    peak = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2

    # Ground depth = median of the bottom 20% of rows.
    h = depth_m.shape[0]
    ground_band = depth_m[int(h * 0.8):, :]
    gvalid = (ground_band > 0.05) & (ground_band < 1.5)
    ground = np.median(ground_band[gvalid]) if gvalid.any() else 1e9

    band = 0.06   # +/- 6 cm around the foot peak
    mask = (np.abs(depth_m - peak) < band) & valid
    mask &= np.abs(depth_m - ground) > 0.015      # drop the floor

    return _largest_blob(mask)


def _yolo_foot_mask(depth_m: np.ndarray, yolo_model) -> np.ndarray | None:
    """
    Backend YOLOv8-seg foot segmentation.

    `yolo_model` is an ultralytics YOLO instance:

        from ultralytics import YOLO
        yolo_model = YOLO("weights/foot_yolov8_seg.pt")

    Returns a boolean mask at the depth-map resolution, or None if no
    confident foot is found.
    """
    h, w = depth_m.shape[:2]
    img = _depth_to_model_input(depth_m)            # HxWx3 uint8

    results = yolo_model.predict(
        img, imgsz=640, conf=0.35, iou=0.5,
        max_det=3, verbose=False)
    if not results:
        return None
    res = results[0]
    if res.masks is None or len(res.masks) == 0:
        return None

    # Single foot per frame -> take the highest-confidence detection.
    confs = res.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))

    # res.masks.data[i] is a float mask at the inference resolution.
    m = res.masks.data[best].cpu().numpy()          # (mh, mw)
    if m.shape != (h, w):
        # Resize (nearest) back to the depth resolution.
        m_img = Image.fromarray((m > 0.5).astype(np.uint8) * 255)
        m_img = m_img.resize((w, h), Image.NEAREST)
        mask = np.asarray(m_img) > 127
    else:
        mask = m > 0.5

    return _largest_blob(mask)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Largest 4-connected component of a boolean mask."""
    from scipy import ndimage
    labelled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labelled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labelled == keep


# ─────────────────────────────────────────────────────────────────────────────
# 4. Point cloud creation
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_cloud(depth_m: np.ndarray, mask: np.ndarray,
                   intr: dict) -> o3d.geometry.PointCloud:
    """Unproject the masked foot pixels into a 3D point cloud (camera frame)."""
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    ys, xs = np.where(mask & (depth_m > 0.05))
    z = depth_m[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Light statistical outlier removal — kills flying depth pixels.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# 5. Registration
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_matrix(quat: list) -> np.ndarray:
    """Attitude quaternion [x,y,z,w] -> 4x4 transform (rotation only)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    return T


def register_all(clouds: list[o3d.geometry.PointCloud],
                  attitudes: list[list],
                  voxel: float = 0.003) -> list[np.ndarray]:
    """
    Register every cloud into the frame of cloud 0.

    Strategy:
      1. seed each pair with the relative device attitude (pose prior)
      2. FPFH + RANSAC global registration to correct the seed
      3. point-to-plane ICP refinement
      4. pose-graph optimisation for global consistency

    Returns a list of 4x4 transforms (cloud_i -> cloud_0 frame).
    """
    n = len(clouds)
    downs, fpfhs = [], []
    for c in clouds:
        d = c.voxel_down_sample(voxel)
        d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            d, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 5, max_nn=100))
        downs.append(d)
        fpfhs.append(f)

    # Pose graph: node 0 is the reference.
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    seeds = [_quat_to_matrix(q) for q in attitudes]
    seed0_inv = np.linalg.inv(seeds[0])

    for i in range(1, n):
        prior = seed0_inv @ seeds[i]   # attitude-based initial guess

        # Global registration to correct the prior.
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            downs[i], downs[0], fpfhs[i], fpfhs[0], True, voxel * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        init = ransac.transformation if ransac.fitness > 0.3 else prior

        # Point-to-plane ICP refinement (this is where 2 mm is won).
        icp = o3d.pipelines.registration.registration_icp(
            clouds[i], clouds[0], voxel * 1.0, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))

        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(icp.transformation)))
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            clouds[i], clouds[0], voxel * 1.5, icp.transformation)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, 0, icp.transformation, info, uncertain=False))

    # Global optimisation.
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=voxel * 1.5,
            edge_prune_threshold=0.25, reference_node=0))

    return [pose_graph.nodes[i].pose for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Surface reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_surface(clouds, transforms,
                         method: str = "poisson") -> o3d.geometry.TriangleMesh:
    """Merge the registered clouds and reconstruct a watertight mesh."""
    merged = o3d.geometry.PointCloud()
    for c, T in zip(clouds, transforms):
        merged += c.transform(T)

    merged = merged.voxel_down_sample(0.001)      # 1 mm
    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=30))
    merged.orient_normals_consistent_tangent_plane(30)

    if method == "poisson":
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            merged, depth=10)
        # Trim low-density (extrapolated) regions.
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
    else:
        # TSDF-style: ball-pivoting for an interpolation-free surface.
        radii = [0.002, 0.004, 0.008]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            merged, o3d.utility.DoubleVector(radii))

    # Cleanup.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = _largest_mesh_component(mesh)
    # Taubin smoothing preserves volume (unlike Laplacian) — important for 2 mm.
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()
    return mesh


def _largest_mesh_component(mesh):
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    keep = int(np.argmax(counts))
    mesh.remove_triangles_by_mask(idx != keep)
    mesh.remove_unreferenced_vertices()
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# 7. Align loaded mesh to the unloaded reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def align_to_unloaded(loaded_mesh: o3d.geometry.TriangleMesh,
                       unloaded_mesh_path: str) -> np.ndarray:
    """
    Register the loaded mesh onto the existing unloaded reconstruction so
    both live in one coordinate frame. ICP on sampled surface points; the
    heel/rear-foot region is the most weight-invariant, so it dominates the
    fit naturally. Returns the 4x4 loaded->unloaded transform.
    """
    unloaded = o3d.io.read_triangle_mesh(unloaded_mesh_path)
    src = loaded_mesh.sample_points_uniformly(40000)
    dst = unloaded.sample_points_uniformly(40000)
    for p in (src, dst):
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        src, dst, 0.01, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return result.transformation


# ─────────────────────────────────────────────────────────────────────────────
# 8. Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_loaded_scan(zip_path: str,
                            out_mesh_path: str,
                            unloaded_mesh_path: str | None = None,
                            yolo_model=None) -> dict:
    """
    Full pipeline. Returns a report dict with per-stage diagnostics.
    Raise/inspect the report's `rmse_*` fields to verify the 2 mm target.
    """
    angles = load_scan_zip(zip_path)
    if len(angles) < 2:
        raise ValueError(f"Need >=2 angles, got {len(angles)}")

    clouds, attitudes = [], []
    for a in angles:
        fused = fuse_burst(a.depth_frames)

        # Mask priority: on-device stored mask -> backend YOLO -> geometric.
        mask = fuse_masks(a.mask_frames)
        if mask is None or not mask.any() or mask.shape != fused.shape:
            mask = segment_foot(fused, yolo_model=yolo_model)

        pcd = depth_to_cloud(fused, mask, a.intrinsics)
        clouds.append(pcd)
        attitudes.append(a.attitude_quat)

    transforms = register_all(clouds, attitudes)
    mesh = reconstruct_surface(clouds, transforms, method="poisson")

    report: dict = {
        "angle_count": len(angles),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }

    if unloaded_mesh_path:
        T = align_to_unloaded(mesh, unloaded_mesh_path)
        mesh.transform(T)
        report["aligned_to_unloaded"] = True

    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    report["mesh_path"] = out_mesh_path
    return report


if __name__ == "__main__":
    import sys
    rep = reconstruct_loaded_scan(
        zip_path=sys.argv[1],
        out_mesh_path=sys.argv[2],
        unloaded_mesh_path=sys.argv[3] if len(sys.argv) > 3 else None,
    )
    print(json.dumps(rep, indent=2))