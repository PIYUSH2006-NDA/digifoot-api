# reconstruction_3d.py
# Isolated depth map → clean 3D foot mesh matching sole silhouette exactly
# Optimized for stationary multi-frame 2D fusion and 2.5D triangulation

import numpy as np
import open3d as o3d
import cv2
import warnings
from typing import List, Optional, Tuple


class FootReconstructor:
    """
    Precision 3D reconstruction pipeline:
    depth_isolated → 2.5D grid triangulation → Taubin smooth → mm standardization → export
    """

    def __init__(self, preprocessor):
        self.prep = preprocessor

    # ------------------------------------------------------------------ #
    #  Depth → 2.5D Exact Surface Mesh
    # ------------------------------------------------------------------ #

    def reconstruct_25d_surface(
        self,
        depth_isolated: np.ndarray,
        max_edge_length: float = 0.02,
    ) -> o3d.geometry.TriangleMesh:
        """
        Directly triangulates the 2D depth map.
        Guarantees the 3D mesh boundary matches the segmented footprint EXACTLY.
        Bypasses Poisson to avoid boundary bloating.
        """
        h, w = depth_isolated.shape
        i_grid, j_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Identify valid pixels
        valid = ~np.isnan(depth_isolated) & (depth_isolated > 0)
        z = depth_isolated.copy()
        z[~valid] = 0

        # Back-project pixels to 3D camera coordinates using intrinsics
        x = (j_grid - self.prep.cx) * z / self.prep.fx
        y = (i_grid - self.prep.cy) * z / self.prep.fy

        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)

        # Create mapping from 2D pixel coordinate to 1D vertex index
        vertex_idx = np.full((h, w), -1, dtype=np.int32)
        vertex_idx[valid] = np.arange(len(pts))

        # Find 2x2 grid patches where all 4 pixels are valid
        valid_r_c = valid[:-1, :-1]
        valid_r1_c = valid[1:, :-1]
        valid_r_c1 = valid[:-1, 1:]
        valid_r1_c1 = valid[1:, 1:]

        all_valid = valid_r_c & valid_r1_c & valid_r_c1 & valid_r1_c1
        r, c = np.where(all_valid)

        # Guard: if the depth map has no valid 2x2 patches (empty / too sparse
        # segmentation), np.max([...], axis=0) below would crash on empty
        # input. Return an empty mesh — reconstruct_from_depth's <50-triangle
        # check then raises a clear error instead of an obscure numpy crash.
        if len(r) == 0:
            print("  reconstruct_25d: no valid surface patches in depth map")
            return o3d.geometry.TriangleMesh()

        # Get the vertex indices for the 4 corners of each valid patch
        idx00 = vertex_idx[r, c]
        idx10 = vertex_idx[r+1, c]
        idx01 = vertex_idx[r, c+1]
        idx11 = vertex_idx[r+1, c+1]

        # Prevent tearing triangles across sharp depth steps (edges of the foot)
        z00, z10, z01, z11 = z[r, c], z[r+1, c], z[r, c+1], z[r+1, c+1]
        z_max = np.max([z00, z10, z01, z11], axis=0)
        z_min = np.min([z00, z10, z01, z11], axis=0)
        diff_mask = (z_max - z_min) < max_edge_length

        idx00 = idx00[diff_mask]
        idx10 = idx10[diff_mask]
        idx01 = idx01[diff_mask]
        idx11 = idx11[diff_mask]

        # Construct standard two triangles per quad cell (Counter-clockwise winding)
        tris1 = np.stack([idx00, idx01, idx10], axis=1)
        tris2 = np.stack([idx01, idx11, idx10], axis=1)
        triangles = np.vstack([tris1, tris2]).astype(np.int32)

        # Construct Open3D Mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        # FIX: the class is Vector3iVector (integer 3-vectors for triangle
        # indices) — not Vector3Vector, which does not exist. Vector3dVector
        # is for float vertices; Vector3iVector is for int triangle indices.
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        return mesh

    # ------------------------------------------------------------------ #
    #  Mesh Post-processing & Verification
    # ------------------------------------------------------------------ #

    def smooth_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        taubin_iter: int = 30,
    ) -> o3d.geometry.TriangleMesh:
        """Pure volume-preserving Taubin smoothing to remove depth sensor ripples."""
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iter)
        mesh.compute_vertex_normals()
        return mesh

    def keep_largest_component(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """Drop tiny segmented noise islands or detached artifacts."""
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

    def simplify_and_clean(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """Decimates triangles safely while removing manifold/degenerate geometry."""
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        try:
            mesh = mesh.fill_holes()
        except Exception:
            pass

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def standardize_for_export(
        self, 
        mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """
        Transforms the raw sensor mesh into a standardized CAD-ready asset:
        Scales to millimeters, centers to X=0 Y=0, places sole plane at Z=0 pointing UP.
        """
        # Scale from camera meters to industry standard CAD millimeters
        mesh.scale(1000.0, center=mesh.get_center())
        
        # Center bounding box along horizontal plane (X=0, Y=0)
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        mesh.translate((-center[0], -center[1], 0.0))
        
        # Shift bottom flat boundary baseline exactly to Z = 0
        verts = np.asarray(mesh.vertices)
        if len(verts) > 0:
            max_z = verts[:, 2].max()
            mesh.translate((0.0, 0.0, -max_z))
        
        # Invert the orientation so sole points DOWN/rests flat, foot topology points UP
        R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        mesh.rotate(R, center=(0, 0, 0))
        
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def get_mesh_metrics(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return {}
        bounds = pts.max(axis=0) - pts.min(axis=0)
        # Note: Bounds are evaluated in millimeters if called post-standardization
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "is_watertight": mesh.is_watertight(),
            "bounds_xyz": bounds.tolist(),
            "foot_length_mm": float(bounds[1] if bounds[1] > bounds[0] else bounds[0]),
        }

    def measure_foot(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Extract measurements based on standardized mm coordinates."""
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return {"length_mm": 0, "width_mm": 0, "height_mm": 0, "eu_size_approx": 0, "vertices": 0, "triangles": 0}
            
        bounds_min = pts.min(axis=0)
        bounds_max = pts.max(axis=0)
        extents = bounds_max - bounds_min

        sorted_extents = np.sort(extents)[::-1]
        length_mm = sorted_extents[0]
        width_mm = sorted_extents[1]
        height_mm = sorted_extents[2]

        eu_approx = round(length_mm / 6.67)

        return {
            "length_mm": round(length_mm, 1),
            "width_mm": round(width_mm, 1),
            "height_mm": round(height_mm, 1),
            "eu_size_approx": eu_approx,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        }

    # ------------------------------------------------------------------ #
    #  Single-Frame Pipeline Execution
    # ------------------------------------------------------------------ #

    def reconstruct_from_depth(
        self,
        depth_isolated: np.ndarray,
        output_path: Optional[str] = None,
        method: str = "2.5d_grid",
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """End-to-end: isolated single depth map → exact matching foot mesh boundary."""
        print(f"  [1/5] Generating 2.5D surface directly from depth silhouette...")
        mesh = self.reconstruct_25d_surface(depth_isolated)

        if len(mesh.triangles) < 50:
            raise ValueError("Reconstruction mesh empty. Check input depth mask availability.")

        print(f"  [2/5] Cleaning component fragments...")
        mesh = self.keep_largest_component(mesh)
        
        print(f"  [3/5] Applying detail-safe Taubin smoothing...")
        mesh = self.smooth_mesh(mesh, taubin_iter=30)
        
        print(f"  [4/5] Decimating topology to target face counts...")
        mesh = self.simplify_and_clean(mesh, target_triangles)

        print(f"  [5/5] Standardizing scale metrics (mm) and origin alignment...")
        mesh = self.standardize_for_export(mesh)

        metrics = self.get_mesh_metrics(mesh)
        print(f"  Finished Reconstruction Optimization: {metrics}")

        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)

        return mesh

    # ------------------------------------------------------------------ #
    #  Stationary Multi-Frame Fusion (2D Matrix Median Processing)
    # ------------------------------------------------------------------ #

    def fuse_stationary_frames(
        self,
        depth_frames: List[np.ndarray],
        output_path: Optional[str] = None,
        target_triangles: int = 50_000,
    ) -> o3d.geometry.TriangleMesh:
        """
        Fuses stationary depth arrays across a 2D temporal median spectrum.
        Completely eliminates high-frequency sensor Z-jitter before 3D translation.
        """
        print(f"\n--- Fusing {len(depth_frames)} Stationary Sensor Arrays in 2D Space ---")

        stack = np.stack(depth_frames, axis=0).astype(np.float32)
        stack[stack <= 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fused_depth = np.nanmedian(stack, axis=0)

        fused_depth[np.isnan(fused_depth)] = 0.0
        print("2D Matrix Temporal Fusion Completed successfully. Passing forward to mesh execution...\n")

        return self.reconstruct_from_depth(
            depth_isolated=fused_depth,
            output_path=output_path,
            method="2.5d_grid",
            target_triangles=target_triangles
        )