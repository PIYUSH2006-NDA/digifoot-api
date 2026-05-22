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
        Non-watertight, good for partial/single-sided scans.

        CHANGED (v7.12): radii reduced from [0.5,1,2,4,8]×avg to
        [0.75,1.5,3]×avg. The large 8×avg ball bridged across real gaps and
        bulged the surface outward — the bloated lump seen in results.
        Smaller balls follow the true surface; tiny holes are fine (the foot
        is an open shell anyway).
        """
        distances = pcd.compute_nearest_neighbor_distance()
        avg = float(np.mean(distances))
        radii = [avg * r for r in [0.75, 1.5, 3.0]]
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

    def remove_nan_vertices(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """
        Remove non-finite (NaN/Inf) vertices. Poisson + Laplacian/Taubin
        smoothing can emit them, which then poisons every downstream
        np.min / bbox / crop operation.
        """
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return mesh
        bad = ~np.isfinite(verts).all(axis=1)
        if bad.any():
            print(f"  Removing {int(bad.sum())} non-finite vertices")
            mesh.remove_vertices_by_mask(bad)
        # Also drop degenerate / unreferenced geometry
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    def keep_largest_component(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """
        Keep ONLY the largest connected component of the mesh.

        Scans pick up floor patches, ankle/leg bits, or the user's hand as
        separate detached blobs (the floating fragments seen in result
        screenshots). The foot is always the biggest connected piece —
        everything else is junk. Drop all but the largest cluster.
        """
        try:
            tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
        except Exception as e:
            print(f"  keep-largest: cluster failed ({e}) — skipping")
            return mesh

        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tri = np.asarray(cluster_n_tri)
        if len(cluster_n_tri) <= 1:
            return mesh   # already a single piece

        largest = int(cluster_n_tri.argmax())
        remove = tri_clusters != largest
        n_removed = int(remove.sum())
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
        print(f"  keep-largest: dropped {len(cluster_n_tri) - 1} junk "
              f"fragment(s), removed {n_removed} triangles")
        return mesh

    def smooth_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        laplacian_iter: int = 8,
        taubin_iter: int = 30,
    ) -> o3d.geometry.TriangleMesh:
        """
        Laplacian + Taubin smoothing to remove scan noise.

        CHANGED (v7.12): laplacian 5→8, taubin 10→30. TrueDepth has ~1-2mm
        per-pixel noise; fused over 18 frames it becomes visible surface
        ripple/lumpiness. Heavier Taubin (volume-preserving, won't shrink the
        foot) rounds it off. Laplacian raised modestly — too much Laplacian
        alone shrinks the mesh, Taubin does the heavy lifting.
        """
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
        """
        Crop mesh to tight bounding box (removes Poisson boundary ghosts).

        FIX (v7.11): guard against empty / all-NaN meshes — Open3D's crop
        throws "AxisAlignedBoundingBox has zero size or wrong bounds" if the
        bbox can't be built. Return the mesh unchanged in that case.
        """
        verts = np.asarray(mesh.vertices)
        if len(verts) < 10:
            print("  bbox-crop: mesh too small — skipping")
            return mesh
        if not np.isfinite(verts).all():
            print("  bbox-crop: ⚠ mesh has non-finite verts — skipping")
            return mesh
        try:
            pts = o3d.geometry.PointCloud()
            pts.points = mesh.vertices
            bbox = pts.get_axis_aligned_bounding_box()
            cropped = mesh.crop(bbox)
            return cropped if len(cropped.vertices) >= 10 else mesh
        except Exception as e:
            print(f"  bbox-crop: ⚠ failed ({e}) — keeping uncropped mesh")
            return mesh

    def crop_foot_depth(
        self,
        mesh: o3d.geometry.TriangleMesh,
        max_foot_depth: float = 0.16,
    ) -> o3d.geometry.TriangleMesh:
        """
        Crop mesh in the Z (camera-depth) axis to just the foot.

        A foot scanned from above is only ~5-9 cm tall. Poisson/fusion
        produces a much deeper mesh because of leg leak + watertight wall
        extrusion. Keep only [z_min, z_min + max_foot_depth].

        FIX (v7.11): mesh vertices can contain NaN (Poisson + smoothing can
        emit them). np.min() on a NaN array returns NaN → invalid crop box →
        0 verts. Use np.nanmin and guard against an all-NaN / empty mesh.
        """
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            print("  Z-crop: mesh already empty — skipping")
            return mesh

        finite = np.isfinite(verts).all(axis=1)
        if not finite.any():
            print("  Z-crop: ⚠ all vertices non-finite — skipping crop")
            return mesh
        vf = verts[finite]

        z_min = float(vf[:, 2].min())
        z_cut = z_min + max_foot_depth
        x_min, y_min = float(vf[:, 0].min()), float(vf[:, 1].min())
        x_max, y_max = float(vf[:, 0].max()), float(vf[:, 1].max())

        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([x_min - 0.01, y_min - 0.01, z_min - 0.005]),
            max_bound=np.array([x_max + 0.01, y_max + 0.01, z_cut]),
        )
        cropped = mesh.crop(crop_box)

        # If the crop somehow emptied the mesh, keep the original — a too-tall
        # mesh is far better than a failed job.
        if len(cropped.vertices) < 10:
            print(f"  Z-crop: ⚠ crop left {len(cropped.vertices)} verts — "
                  f"keeping uncropped mesh")
            return mesh

        print(f"  Z-crop: kept [{z_min:.3f}, {z_cut:.3f}]m "
              f"({len(cropped.vertices)} verts, was {len(verts)})")
        return cropped

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
        mesh = self.remove_nan_vertices(mesh)
        mesh = self.keep_largest_component(mesh)   # drop floating junk
        mesh = self.crop_foot_depth(mesh, max_foot_depth=0.16)
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
        voxel_size: float = 0.0015,
        output_path: Optional[str] = None,
        method: str = "ball_pivot",
        target_triangles: int = 80_000,
        max_foot_depth: float = 0.16,
    ) -> o3d.geometry.TriangleMesh:
        """
        Fuse multiple depth frames → denser point cloud → mesh.

        CHANGED (v7.10): default method is now ball_pivot, NOT poisson.

        WHY: Poisson is a *watertight* reconstruction — it must close the
        volume. A foot scanned from above is a single open surface (top only,
        no bottom/sides). Poisson seals it by extruding the open boundary
        edges toward the camera → those extrusions are the "side walls
        raising too high" the user reported. Ball-pivoting drapes a surface
        over the points and STOPS at boundaries — correct for an open scan.

        Also: Z-crop to foot depth + decimation (was producing 27-58 MB STLs).
        """
        print(f"Fusing {len(depth_frames)} frames (method={method})...")
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
        # CHANGED (v7.13): voxel_size 0.0015 → 0.0022. A larger merge cell
        # averages more points together, and averaging is what cancels
        # random sensor noise — the rippled surface texture. 2.2mm cells
        # still keep toe/arch detail (a toe is ~15mm wide).
        combined = combined.voxel_down_sample(0.0022)
        print(f"  After voxel merge: {len(combined.points)}")

        # Cross-frame cleaning — fused cloud has outliers from every frame
        # (leg/floor leak, flying pixels). Statistical then radius pass.
        combined = self.remove_statistical_outliers(combined, nb_neighbors=30,
                                                    std_ratio=1.5)
        combined = self.remove_radius_outliers(combined, nb_points=12,
                                               radius=0.009)
        print(f"  After cross-frame outlier removal: {len(combined.points)}")
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
            # Ball-pivot can fragment on noisy clouds — fall back to Poisson
            # ONLY if it produced almost nothing.
            if len(mesh.triangles) < 200:
                print(f"  Ball-pivot gave {len(mesh.triangles)} tris — "
                      f"falling back to poisson")
                mesh = self.reconstruct_poisson(combined, depth=9)

        if len(mesh.triangles) < 10:
            raise ValueError(f"Fused reconstruction produced empty mesh")

        mesh = self.smooth_mesh(mesh)
        mesh = self.remove_nan_vertices(mesh)
        mesh = self.keep_largest_component(mesh)   # drop floating junk
        # Z-crop FIRST (removes wall extrusions), then tight bbox.
        mesh = self.crop_foot_depth(mesh, max_foot_depth=max_foot_depth)
        mesh = self.crop_to_bbox(mesh)
        # FIX: decimate — fuse_frames never simplified → 27-58 MB STLs.
        mesh = self.simplify_mesh(mesh, target_triangles)
        print(f"  Final mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

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