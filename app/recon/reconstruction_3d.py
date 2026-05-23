# reconstruction_3d.py
# Isolated depth map → clean 3D foot mesh
# Optimized for stationary multi-frame fusion and volume-preserving detail

import numpy as np
import open3d as o3d
import cv2
import warnings
from typing import List, Optional, Tuple


class FootReconstructor:
    """
    Optimized 3D reconstruction pipeline:
    depth_isolated → point cloud → orient normals → Poisson reconstruct → Taubin smooth → export
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
        """Project isolated depth map → Open3D PointCloud."""
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
            colors = cmap[:, ::-1] / 255.0  # BGR → RGB
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    # ------------------------------------------------------------------ #
    #  Point Cloud Cleaning & Normals
    # ------------------------------------------------------------------ #

    def clean_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply full cleaning pipeline to remove flying pixels."""
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.01)
        return pcd.voxel_down_sample(voxel_size=0.002)

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.015,
        max_nn: int = 30,
    ) -> o3d.geometry.PointCloud:
        """Estimate + orient surface normals strictly toward camera."""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        # Consistent tangent plane can flip normals on open surfaces. 
        # Orienting directly to the camera origin is much safer for Poisson.
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
        Poisson surface reconstruction.
        Linear_fit=True is essential for closing open boundaries smoothly without bloating.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale, linear_fit=True
        )
        dens = np.asarray(densities)
        threshold = np.quantile(dens, density_percentile / 100.0)
        mesh.remove_vertices_by_mask(dens < threshold)
        return mesh

    def reconstruct_ball_pivot(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:
        """Fallback for highly sparse point clouds."""
        distances = pcd.compute_nearest_neighbor_distance()
        avg = float(np.mean(distances))
        radii = [avg * r for r in [1.0, 1.5, 2.0, 3.0]]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        return mesh

    # ------------------------------------------------------------------ #
    #  Mesh Post-processing
    # ------------------------------------------------------------------ #

    def smooth_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        taubin_iter: int = 40,
    ) -> o3d.geometry.TriangleMesh:
        """
        Pure Taubin smoothing. 
        Volume-preserving filter that eliminates sensor ripple without shrinking toes.
        """
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iter)
        mesh.compute_vertex_normals()
        return mesh

    def keep_largest_component(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """Keep ONLY the largest connected component of the mesh."""
        try:
            tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
        except Exception:
            return mesh

        cluster_n_tri = np.asarray(cluster_n_tri)
        if len(cluster_n_tri) <= 1:
            return mesh

        largest = int(cluster_n_tri.argmax())
        remove = np.asarray(tri_clusters) != largest
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
        return mesh

    def crop_foot_depth(
        self,
        mesh: o3d.geometry.TriangleMesh,
        max_foot_depth: float = 0.12,
    ) -> o3d.geometry.TriangleMesh:
        """Crop mesh in the Z (camera-depth) axis to remove Poisson back-wall extrusions."""
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return mesh
        
        finite = np.isfinite(verts).all(axis=1)
        if not finite.any():
            return mesh
        
        finite_verts = verts[finite]
        z_min = float(finite_verts[:, 2].min())
        z_cut = z_min + max_foot_depth
        
        x_min, y_min = float(finite_verts[:, 0].min()), float(finite_verts[:, 1].min())
        x_max, y_max = float(finite_verts[:, 0].max()), float(finite_verts[:, 1].max())

        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([x_min - 0.02, y_min - 0.02, z_min - 0.01]),
            max_bound=np.array([x_max + 0.02, y_max + 0.02, z_cut]),
        )
        cropped = mesh.crop(crop_box)
        return cropped if len(cropped.vertices) >= 50 else mesh

    def simplify_and_clean(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """Safe decimation and normal computation."""
        # Clean degenerate geometry before and after decimation
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # Open3D built-in hole filling for tiny gaps left by decimation
        try:
            mesh = mesh.fill_holes()
        except Exception:
            pass

        mesh.compute_vertex_normals()
        return mesh

    def get_mesh_metrics(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Return basic mesh quality metrics."""
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return {}
        bounds = pts.max(axis=0) - pts.min(axis=0)
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "is_watertight": mesh.is_watertight(),
            "bounds_xyz_m": bounds.tolist(),
            "foot_length_mm": bounds[2] * 1000 if bounds[2] > bounds[0] else bounds[0] * 1000,
        }

    # ------------------------------------------------------------------ #
    #  Full Pipeline (Single Frame)
    # ------------------------------------------------------------------ #

    def reconstruct_from_depth(
        self,
        depth_isolated: np.ndarray,
        output_path: Optional[str] = None,
        method: str = "poisson",
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """End-to-end: isolated depth map → clean foot mesh."""
        
        print(f"  [1/6] Depth → PointCloud")
        pcd = self.depth_to_pcd(depth_isolated, colorize=True)
        if len(pcd.points) < 100:
            raise ValueError("Too few 3D points — depth frame had almost no valid foot pixels")

        print(f"  [2/6] Cleaning outliers")
        pcd = self.clean_pcd(pcd)
        if len(pcd.points) < 50:
            raise ValueError("Too few points after cleaning")

        print(f"  [3/6] Estimating normals")
        pcd = self.estimate_normals(pcd)

        print(f"  [4/6] Surface reconstruction ({method})")
        if method == "poisson":
            mesh = self.reconstruct_poisson(pcd)
            if len(mesh.triangles) < 50:
                print("        Poisson failed — falling back to ball-pivoting")
                mesh = self.reconstruct_ball_pivot(pcd)
        elif method == "ball_pivot":
            mesh = self.reconstruct_ball_pivot(pcd)
        else:
            raise ValueError(f"Unknown method: {method}")

        if len(mesh.triangles) < 10:
            raise ValueError("Reconstruction produced an empty mesh")

        print(f"  [5/6] Smoothing + simplification")
        mesh = self.keep_largest_component(mesh)
        mesh = self.crop_foot_depth(mesh, max_foot_depth=0.12)
        
        # Heavy Taubin smooth to eliminate sensor ripple
        mesh = self.smooth_mesh(mesh, taubin_iter=40)
        
        # Decimate and clean degenerate triangles
        mesh = self.simplify_and_clean(mesh, target_triangles)

        metrics = self.get_mesh_metrics(mesh)
        print(f"  [6/6] Done. {metrics}")

        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"        Saved: {output_path}")

        return mesh

    # ------------------------------------------------------------------ #
    #  Stationary Multi-Frame Fusion (2D Temporal Fusion)
    # ------------------------------------------------------------------ #

    def fuse_stationary_frames(
        self,
        depth_frames: List[np.ndarray],
        output_path: Optional[str] = None,
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """
        For stationary cameras: Fuses depth maps in 2D using a temporal median.
        This completely eliminates sensor Z-jitter and flying pixels BEFORE 
        creating a single 3D point cloud, resulting in a glass-smooth surface.
        """
        print(f"\n--- Fusing {len(depth_frames)} stationary frames in 2D ---")

        # Stack all depth frames into a single 3D array: (Time, Height, Width)
        stack = np.stack(depth_frames, axis=0).astype(np.float32)

        # Mask out invalid pixels (0 or negative) with NaN 
        stack[stack <= 0] = np.nan

        # Compute the median depth per pixel across time.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fused_depth = np.nanmedian(stack, axis=0)

        # Restore NaN values back to 0.0 for the background
        fused_depth[np.isnan(fused_depth)] = 0.0

        print("2D Temporal Fusion complete. Generating single clean mesh...\n")

        # Pass this single, ultra-clean depth map into the standard pipeline.
        return self.reconstruct_from_depth(
            depth_isolated=fused_depth,
            output_path=output_path,
            method="poisson",
            target_triangles=target_triangles
        )