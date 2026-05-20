# reconstruction_3d.py
# Isolated depth map → clean 3D foot mesh
# Supports: single-frame and multi-frame fusion

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Optional, Tuple


class FootReconstructor:
    """
    Full 3D reconstruction pipeline:
      depth_isolated → point cloud → clean → reconstruct mesh → smooth → export
    """

    def __init__(self, preprocessor):
        self.prep = preprocessor

    # ------------------------------------------------------------------ #
    #  Depth → Point Cloud
    # ------------------------------------------------------------------ #

    def depth_to_pcd(
        self,
        depth_isolated: np.ndarray,
        colorize: bool = True,
    ) -> o3d.geometry.PointCloud:
        """
        Project isolated depth map → Open3D PointCloud.
        Uses pinhole camera model (fx, fy, cx, cy from preprocessor).
        """
        h, w = depth_isolated.shape
        i_grid, j_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        valid = ~np.isnan(depth_isolated) & (depth_isolated > 0)
        z = depth_isolated[valid]
        x = (j_grid[valid] - self.prep.cx) * z / self.prep.fx
        y = (i_grid[valid] - self.prep.cy) * z / self.prep.fy

        pts = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if colorize:
            norm_z = (z - z.min()) / (z.max() - z.min() + 1e-6)
            cmap = cv2.applyColorMap(
                (norm_z * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            ).squeeze()
            colors = cmap[:, ::-1] / 255.0  # BGR → RGB, [0,1]
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    # ------------------------------------------------------------------ #
    #  Point Cloud Cleaning
    # ------------------------------------------------------------------ #

    def remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> o3d.geometry.PointCloud:
        """Remove flying pixels and outlier points."""
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return pcd.select_by_index(ind)

    def remove_radius_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_points: int = 16,
        radius: float = 0.01,
    ) -> o3d.geometry.PointCloud:
        """Remove isolated sparse points."""
        _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return pcd.select_by_index(ind)

    def voxel_downsample(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.002,
    ) -> o3d.geometry.PointCloud:
        """Voxel downsampling for uniform point density."""
        return pcd.voxel_down_sample(voxel_size)

    def clean_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply full cleaning pipeline."""
        pcd = self.remove_statistical_outliers(pcd)
        pcd = self.remove_radius_outliers(pcd)
        pcd = self.voxel_downsample(pcd, voxel_size=0.002)
        return pcd

    # ------------------------------------------------------------------ #
    #  Normals
    # ------------------------------------------------------------------ #

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.02,
        max_nn: int = 30,
    ) -> o3d.geometry.PointCloud:
        """Estimate + orient surface normals (required for Poisson)."""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        # Orient toward camera origin
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        return pcd

    # ------------------------------------------------------------------ #
    #  Mesh Reconstruction
    # ------------------------------------------------------------------ #

    def reconstruct_poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
        scale: float = 1.1,
        density_percentile: float = 5.0,
    ) -> o3d.geometry.TriangleMesh:
        """
        Poisson surface reconstruction → watertight mesh.
        Removes low-density boundary artifacts automatically.
        Best for smooth, complete foot surfaces.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale, linear_fit=False
        )
        # Remove low-density boundary vertices (reconstruction artifacts)
        dens = np.asarray(densities)
        threshold = np.percentile(dens, density_percentile)
        remove_mask = dens < threshold
        mesh.remove_vertices_by_mask(remove_mask)
        return mesh

    def reconstruct_ball_pivot(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:
        """
        Ball-pivoting reconstruction.
        Faster than Poisson, non-watertight but good for partial scans.
        """
        distances = pcd.compute_nearest_neighbor_distance()
        avg = np.mean(distances)
        radii = [avg * r for r in [0.5, 1.0, 2.0, 4.0, 8.0]]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        return mesh

    def reconstruct_alpha_shape(
        self,
        pcd: o3d.geometry.PointCloud,
        alpha: float = 0.03,
    ) -> o3d.geometry.TriangleMesh:
        """
        Alpha shape reconstruction.
        Good for concave shapes (arch of foot).
        """
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
        return mesh

    # ------------------------------------------------------------------ #
    #  Mesh Post-processing
    # ------------------------------------------------------------------ #

    def smooth_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        laplacian_iter: int = 5,
        taubin_iter: int = 10,
    ) -> o3d.geometry.TriangleMesh:
        """Laplacian + Taubin smoothing to remove scan noise."""
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iter)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iter)
        mesh.compute_vertex_normals()
        return mesh

    def simplify_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """Decimate mesh to target triangle count (mobile-friendly)."""
        current = len(mesh.triangles)
        if current <= target_triangles:
            return mesh
        mesh = mesh.simplify_quadric_decimation(target_triangles)
        mesh.compute_vertex_normals()
        return mesh

    def crop_to_bbox(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """Crop mesh to tight bounding box (removes Poisson boundary ghosts)."""
        pts = o3d.geometry.PointCloud()
        pts.points = mesh.vertices
        bbox = pts.get_axis_aligned_bounding_box()
        return mesh.crop(bbox)

    def fill_holes(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Fill small holes in mesh (useful for arch underside gaps)."""
        mesh = mesh.fill_holes()
        return mesh

    def get_mesh_metrics(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Return basic mesh quality metrics."""
        pts = np.asarray(mesh.vertices)
        bounds = pts.max(axis=0) - pts.min(axis=0)
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "is_watertight": mesh.is_watertight(),
            "bounds_xyz_m": bounds.tolist(),
            "foot_length_mm": bounds[2] * 1000 if bounds[2] > bounds[0] else bounds[0] * 1000,
        }

    # ------------------------------------------------------------------ #
    #  Full Pipeline
    # ------------------------------------------------------------------ #

    def reconstruct_from_depth(
        self,
        depth_isolated: np.ndarray,
        output_path: Optional[str] = None,
        method: str = "poisson",
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """
        End-to-end: isolated depth map → clean foot mesh.

        Args:
            depth_isolated: float32 depth (NaN for non-foot pixels)
            output_path:    save .obj / .ply / .stl if provided
            method:         'poisson' | 'ball_pivot' | 'alpha_shape'
            target_triangles: mesh simplification target

        Returns:
            Open3D TriangleMesh
        """
        print(f"  [1/6] Depth → PointCloud")
        pcd = self.depth_to_pcd(depth_isolated, colorize=True)
        print(f"        {len(pcd.points)} raw points")
        if len(pcd.points) < 100:
            raise ValueError(f"Too few 3D points ({len(pcd.points)}) — "
                             f"depth frame had almost no valid foot pixels")

        print(f"  [2/6] Cleaning outliers")
        pcd = self.clean_pcd(pcd)
        print(f"        {len(pcd.points)} clean points")
        if len(pcd.points) < 50:
            raise ValueError(f"Too few points after cleaning ({len(pcd.points)})")

        print(f"  [3/6] Estimating normals")
        pcd = self.estimate_normals(pcd)

        print(f"  [4/6] Surface reconstruction ({method})")
        if method == "poisson":
            mesh = self.reconstruct_poisson(pcd)
            # Poisson can return an empty mesh on sparse/partial clouds.
            # Fall back to ball-pivoting which handles open surfaces better.
            if len(mesh.triangles) < 50:
                print(f"        Poisson gave {len(mesh.triangles)} tris — "
                      f"falling back to ball-pivoting")
                mesh = self.reconstruct_ball_pivot(pcd)
        elif method == "ball_pivot":
            mesh = self.reconstruct_ball_pivot(pcd)
        elif method == "alpha_shape":
            mesh = self.reconstruct_alpha_shape(pcd)
        else:
            raise ValueError(f"Unknown method: {method}")

        if len(mesh.triangles) < 10:
            raise ValueError(f"Reconstruction produced an empty mesh "
                             f"({len(mesh.triangles)} triangles)")

        print(f"  [5/6] Smoothing + simplification")
        mesh = self.smooth_mesh(mesh)
        mesh = self.crop_to_bbox(mesh)
        mesh = self.simplify_mesh(mesh, target_triangles)

        metrics = self.get_mesh_metrics(mesh)
        print(f"  [6/6] Done. {metrics}")

        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"        Saved: {output_path}")

        return mesh

    # ------------------------------------------------------------------ #
    #  Multi-Frame Fusion
    # ------------------------------------------------------------------ #

    def fuse_frames(
        self,
        depth_frames: List[np.ndarray],
        voxel_size: float = 0.001,
        output_path: Optional[str] = None,
        method: str = "poisson",
    ) -> o3d.geometry.TriangleMesh:
        """
        Fuse multiple depth frames → denser point cloud → mesh.
        Assumes minor camera motion between frames for richer coverage.

        Typical use: capture 5–15 frames with slight tilt/rotation,
        then call this for a more complete foot reconstruction.
        """
        print(f"Fusing {len(depth_frames)} frames...")
        combined = o3d.geometry.PointCloud()

        for i, d in enumerate(depth_frames):
            pcd_i = self.depth_to_pcd(d)
            pcd_i = self.remove_statistical_outliers(pcd_i)
            combined += pcd_i
            print(f"  Frame {i+1}: {len(pcd_i.points)} pts")

        print(f"  Total before fusion downsample: {len(combined.points)}")
        if len(combined.points) < 100:
            raise ValueError(f"Fused cloud too sparse ({len(combined.points)} pts) "
                             f"— foot segmentation isolated almost nothing")
        combined = combined.voxel_down_sample(voxel_size)
        print(f"  After voxel merge: {len(combined.points)}")
        combined = self.estimate_normals(combined)

        print("  Reconstructing fused mesh...")
        if method == "poisson":
            mesh = self.reconstruct_poisson(combined, depth=10)
            if len(mesh.triangles) < 50:
                print(f"  Poisson gave {len(mesh.triangles)} tris — "
                      f"falling back to ball-pivoting")
                mesh = self.reconstruct_ball_pivot(combined)
        else:
            mesh = self.reconstruct_ball_pivot(combined)

        if len(mesh.triangles) < 10:
            raise ValueError(f"Fused reconstruction produced empty mesh")

        mesh = self.smooth_mesh(mesh)
        mesh = self.crop_to_bbox(mesh)

        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"  Saved fused mesh: {output_path}")

        return mesh

    # ------------------------------------------------------------------ #
    #  Measurements
    # ------------------------------------------------------------------ #

    def measure_foot(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """
        Extract key foot measurements from mesh.
        Assumes foot axis aligned with Z or X axis.
        """
        pts = np.asarray(mesh.vertices)
        bounds_min = pts.min(axis=0)
        bounds_max = pts.max(axis=0)
        extents = bounds_max - bounds_min

        # Sort extents: length = max, width = mid, height = min
        sorted_extents = np.sort(extents)[::-1]
        length_mm = sorted_extents[0] * 1000
        width_mm = sorted_extents[1] * 1000
        height_mm = sorted_extents[2] * 1000

        # EU shoe size estimate (very rough, for demonstration)
        # EU size ≈ foot_length_mm * (3/2) * (10/25.4) ... simplified
        eu_approx = round(length_mm / 6.67)

        return {
            "length_mm": round(length_mm, 1),
            "width_mm": round(width_mm, 1),
            "height_mm": round(height_mm, 1),
            "eu_size_approx": eu_approx,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        }