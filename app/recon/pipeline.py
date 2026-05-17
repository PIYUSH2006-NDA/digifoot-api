"""
backend/app/recon/pipeline.py
Full reconstruction pipeline. ~1mm accuracy target.

Flow:
  load frames → bilateral filter → unproject → ICP refine poses →
  TSDF fuse @ 1mm → marching cubes → outlier removal →
  Poisson surface → ML refine (PointNet++) → remesh → UV → texture → OBJ

Also: measurement_pipeline() for foot dimensions.
"""
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from .ml_refine import refine_with_pointnet2
from .obj_writer import write_obj_rhino
from .uv_bake import unwrap_and_bake
from .measurements import extract_measurements


# ─────────────────────── helpers ───────────────────────

def _load_frames(raw_dir: Path) -> tuple[list[dict], dict]:
    """Load depth .npy, rgb .png, poses.json, intrinsics.json."""
    poses_raw = json.loads((raw_dir / "poses.json").read_text())
    intr = json.loads((raw_dir / "intrinsics.json").read_text())
    # normalize key types
    intr = {k: float(v) for k, v in intr.items()}

    depths = sorted(raw_dir.glob("depth_*.npy"))
    rgbs = sorted(raw_dir.glob("rgb_*.png"))
    n = min(len(depths), len(rgbs), len(poses_raw))

    frames = []
    for i in range(n):
        z = np.load(depths[i]).astype(np.float32)
        # bilateral: preserve edges, reduce noise
        z = cv2.bilateralFilter(z, d=5, sigmaColor=0.02, sigmaSpace=5.0)
        img = cv2.imread(str(rgbs[i]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            h, w = z.shape
            img = np.full((h, w, 3), 128, dtype=np.uint8)

        pose = np.array(poses_raw[i], dtype=np.float64).reshape(4, 4)
        frames.append({"depth": z, "rgb": img, "pose": pose})

    return frames, intr


def _unproject_frame(z: np.ndarray, intr: dict,
                     pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth map → world-space points + colors stub."""
    h, w = z.shape
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    valid = (z > 0.03) & (z < 0.65)
    X = (xs[valid] - cx) * z[valid] / fx
    Y = (ys[valid] - cy) * z[valid] / fy
    Z = z[valid]
    pts_cam = np.stack([X, Y, Z], axis=-1)                    # [M, 3]
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts_cam, ones], axis=-1)           # [M, 4]
    world = (pose @ pts_h.T).T[:, :3]                          # [M, 3]
    return world.astype(np.float32), valid


# ─────────────────────── ICP pose refinement ───────────────────────

def _build_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    return pcd


def _refine_poses_icp(frames: list[dict], intr: dict) -> list[dict]:
    """Multi-scale point-to-plane ICP between consecutive frames."""
    refined_poses = [frames[0]["pose"].copy()]
    prev_pcd = None

    for i in range(len(frames)):
        pts, _ = _unproject_frame(frames[i]["depth"], intr, np.eye(4))
        pcd = _build_pcd(pts)
        if prev_pcd is not None:
            init = np.linalg.inv(refined_poses[-1]) @ frames[i]["pose"]
            for vox in [0.01, 0.005, 0.002]:
                a = pcd.voxel_down_sample(vox)
                b = prev_pcd.voxel_down_sample(vox)
                a.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=vox * 3, max_nn=30))
                b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=vox * 3, max_nn=30))
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50)
                reg = o3d.pipelines.registration.registration_icp(
                    a, b, vox * 2.0, init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria)
                init = reg.transformation
            refined_poses.append(refined_poses[-1] @ init)
        else:
            refined_poses.append(frames[i]["pose"].copy())
        prev_pcd = pcd

    for f, p in zip(frames, refined_poses[1:]):
        f["pose"] = p
    return frames


# ─────────────────────── TSDF fusion ───────────────────────

def _tsdf_fuse(frames: list[dict], intr: dict,
               voxel: float = 0.001) -> o3d.geometry.TriangleMesh:
    """Scalable TSDF volume @ 1mm voxel."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intr_o3d = o3d.camera.PinholeCameraIntrinsic(
        int(intr["w"]), int(intr["h"]),
        intr["fx"], intr["fy"], intr["cx"], intr["cy"])

    for f in frames:
        rgb_o3d = o3d.geometry.Image(f["rgb"].astype(np.uint8))
        # depth in meters → Open3D expects mm-scale uint16 with depth_scale
        depth_mm = (f["depth"] * 1000.0).astype(np.uint16)
        d_o3d = o3d.geometry.Image(depth_mm)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, d_o3d,
            depth_scale=1000.0,
            depth_trunc=0.65,
            convert_rgb_to_intensity=False)
        extrinsic = np.linalg.inv(f["pose"])
        vol.integrate(rgbd, intr_o3d, extrinsic)

    return vol.extract_triangle_mesh()


# ─────────────────────── mesh cleaning ───────────────────────

def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # statistical outlier removal on point cloud, then keep valid triangles
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    mesh = mesh.select_by_index(inlier_idx)

    # largest connected component
    tri_clusters, cluster_n_tri, cluster_area = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_area = np.asarray(cluster_area)
    if len(cluster_area) > 0:
        keep_mask = tri_clusters == cluster_area.argmax()
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()

    # Taubin smooth (volume-preserving)
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=10, lambda_filter=0.5, mu=-0.53)
    mesh.compute_vertex_normals()
    return mesh


# ─────────────────────── Poisson surface recon ───────────────────────

def _poisson_recon(mesh: o3d.geometry.TriangleMesh,
                   depth: int = 10,
                   density_quantile: float = 0.02
                   ) -> o3d.geometry.TriangleMesh:
    """Poisson surface from cleaned mesh samples."""
    pcd = mesh.sample_points_poisson_disk(number_of_points=150_000)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.005, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    p_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    p_mesh.remove_vertices_by_mask(densities < threshold)
    p_mesh.compute_vertex_normals()
    return p_mesh


# ─────────────────────── remesh ───────────────────────

def _remesh(mesh: o3d.geometry.TriangleMesh,
            target_tris: int = 256_000) -> o3d.geometry.TriangleMesh:
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_tris)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


# ═══════════════════════ main pipeline ═══════════════════════

def recon_pipeline(job_dir: Path, mode: str = "truedepth") -> dict:
    """
    Full reconstruction. Returns {"mesh": Path, "cloud": Path}.
    mode: "truedepth" | "lidar" | "photogrammetry"
    """
    t0 = time.time()
    raw = job_dir / "raw"

    # ── photogrammetry: assume pre-fused mesh dropped by iOS ──
    if mode == "photogrammetry":
        candidates = list(raw.glob("*.obj")) + list(raw.glob("*.ply"))
        if candidates:
            mesh = o3d.io.read_triangle_mesh(str(candidates[0]))
        else:
            raise FileNotFoundError("no mesh file found for photogrammetry mode")
    else:
        # ── depth-based: TrueDepth / LiDAR ──
        frames, intr = _load_frames(raw)
        if len(frames) < 10:
            raise ValueError(f"too few frames: {len(frames)} (need ≥10)")

        print(f"[recon] loaded {len(frames)} frames")

        # step 1: ICP refine poses
        frames = _refine_poses_icp(frames, intr)
        print(f"[recon] ICP done")

        # step 2: TSDF fuse
        voxel = 0.001 if mode == "lidar" else 0.0015     # LiDAR finer
        mesh = _tsdf_fuse(frames, intr, voxel=voxel)
        print(f"[recon] TSDF done: {len(mesh.vertices)} verts")

    # step 3: clean
    mesh = _clean_mesh(mesh)
    print(f"[recon] clean: {len(mesh.vertices)} verts")

    # step 4: Poisson surface
    mesh = _poisson_recon(mesh, depth=10)
    print(f"[recon] Poisson: {len(mesh.vertices)} verts")

    # step 5: second clean pass
    mesh = _clean_mesh(mesh)

    # step 6: ML refine (PointNet++ displacement + normal correction)
    mesh = refine_with_pointnet2(mesh)
    print(f"[recon] ML refine done")

    # step 7: remesh to target density
    mesh = _remesh(mesh, target_tris=256_000)
    print(f"[recon] remesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # step 8: save intermediate cloud
    cloud = mesh.sample_points_uniformly(50_000)
    cloud_path = job_dir / "cloud.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud)

    # step 9: UV unwrap + texture bake
    mesh, tex_png = unwrap_and_bake(mesh, raw)
    print(f"[recon] UV + texture done")

    # step 10: write OBJ (Rhino-compatible, matches Lucas file)
    obj_path = job_dir / "foot.obj"
    write_obj_rhino(mesh, tex_png, obj_path, name="foot")

    dt = time.time() - t0
    print(f"[recon] DONE in {dt:.1f}s → {obj_path}")

    return {"mesh": obj_path, "cloud": cloud_path}


# ═══════════════════════ measurement wrapper ═══════════════════════

def measurement_pipeline(mesh_path: Path) -> dict:
    """Load mesh, extract clinical measurements."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return extract_measurements(mesh)
