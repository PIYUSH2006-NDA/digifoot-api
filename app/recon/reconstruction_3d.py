# reconstruction_3d.py  (v8 — Professional single-frame reconstruction)
#
# Replaces the direct pixel-grid triangulation pipeline.
#
# Pipeline:
#   1. Refined depth + smoothed silhouette  (done in DepthPreprocessor)
#   2. Back-project to oriented point cloud (with proper normals)
#   3. Screened Poisson reconstruction       (curvature-adaptive, smooth)
#   4. Density-based crop                    (kill Poisson boundary bloat)
#   5. Clip to foot silhouette               (mesh boundary == foot outline)
#   6. Extrude to flat base                  (closed, watertight solid)
#   7. Curvature-adaptive smoothing          (preserves toes / heel / arch)
#   8. Decimate + manifold repair
#   9. Standardize to mm, CAD-ready orientation
#  10. Write STL

import numpy as np
import open3d as o3d
import cv2
import warnings
from typing import List, Optional, Tuple


class FootReconstructor:
    """
    Single-frame, depth-only foot reconstruction for plantar (sole) scans.

    Produces a closed solid mesh of the sole surface extruded to a flat base.
    Watertight, manifold, CAD-ready in millimeters.
    """

    def __init__(self, preprocessor):
        self.prep = preprocessor

    # ====================================================================
    # STAGE A — Depth -> oriented point cloud
    # ====================================================================

    def _depth_to_oriented_pcd(
        self,
        depth_isolated: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Back-project the isolated depth map and compute proper surface normals.

        Critical: normals are estimated from depth gradients (analytical, fast,
        and always correct for a heightmap) — far more reliable than k-NN
        normal estimation, which fails on noisy data near boundaries.
        """
        h, w = depth_isolated.shape
        valid = ~np.isnan(depth_isolated) & (depth_isolated > 0)
        if valid.sum() < 200:
            return o3d.geometry.PointCloud()

        i_grid, j_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z = depth_isolated.astype(np.float32)
        # 3D back-projection (pinhole)
        x = (j_grid - self.prep.cx) * z / self.prep.fx
        y = (i_grid - self.prep.cy) * z / self.prep.fy

        # Analytical normals from depth gradient: n = normalize(-dz/dx, -dz/dy, 1)
        # Use Scharr (better gradient accuracy than Sobel for normals)
        z_safe = np.where(valid, z, 0.0)
        gx = cv2.Scharr(z_safe, cv2.CV_32F, 1, 0) / (3.0 * self.prep.fx)
        gy = cv2.Scharr(z_safe, cv2.CV_32F, 0, 1) / (3.0 * self.prep.fy)
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen

        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
        nrm = np.stack([nx[valid], ny[valid], nz[valid]], axis=1)

        # Orient normals toward camera (Z=0). For a foot above a phone on the
        # floor, the sole faces the camera, so normals should have negative z.
        flip = nrm[:, 2] > 0
        nrm[flip] *= -1.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
        return pcd

    # ====================================================================
    # STAGE B — Poisson reconstruction with density crop
    # ====================================================================

    def _poisson_with_density_crop(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
        density_percentile: float = 6.0,
    ) -> o3d.geometry.TriangleMesh:
        """
        Screened Poisson with density-driven boundary trimming.

        Poisson produces watertight, curvature-adaptive meshes — toe tips get
        more triangles than the flat midfoot. The classic flaw on open
        surfaces is boundary bloat (the algorithm fills "outside" the actual
        scan with low-density extrapolation); we remove those vertices by
        density percentile, leaving the real surface.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh, densities = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, scale=1.05, linear_fit=True
                )
            )
        densities = np.asarray(densities)
        if len(densities):
            cutoff = np.quantile(densities, density_percentile / 100.0)
            mesh.remove_vertices_by_mask(densities < cutoff)
        mesh.remove_unreferenced_vertices()
        return mesh

    # ====================================================================
    # STAGE C — Clip mesh boundary to true foot silhouette
    # ====================================================================

    def _clip_to_silhouette(
        self,
        mesh: o3d.geometry.TriangleMesh,
        mask: Optional[np.ndarray],
        depth_shape: Tuple[int, int],
    ) -> o3d.geometry.TriangleMesh:
        """
        Drop mesh vertices whose camera projection falls outside the foot
        silhouette mask. This makes the mesh boundary exactly follow the
        (smoothed) foot outline — no Poisson overshoot, no rectangular fringe.
        """
        if mask is None:
            return mesh
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return mesh

        h, w = depth_shape
        # Project vertices back to pixel coords
        z = verts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            j = (verts[:, 0] * self.prep.fx) / z + self.prep.cx
            i = (verts[:, 1] * self.prep.fy) / z + self.prep.cy
        ji = np.stack([j, i], axis=1)

        # Dilate the mask slightly so boundary triangles aren't clipped flush
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dil = cv2.dilate(mask, k, iterations=1)

        inside = np.zeros(len(verts), dtype=bool)
        valid = (z > 0) & np.isfinite(j) & np.isfinite(i)
        jj = np.clip(j[valid].astype(np.int32), 0, w - 1)
        ii = np.clip(i[valid].astype(np.int32), 0, h - 1)
        inside[valid] = mask_dil[ii, jj] > 0

        if not inside.any():
            return mesh
        mesh.remove_vertices_by_mask(~inside)
        mesh.remove_unreferenced_vertices()
        return mesh

    # ====================================================================
    # STAGE D — Curvature-adaptive smoothing
    # ====================================================================

    def _adaptive_smooth(
        self,
        mesh: o3d.geometry.TriangleMesh,
        base_iter: int = 18,
        light_iter: int = 6,
    ) -> o3d.geometry.TriangleMesh:
        """
        Two-pass smoothing that preserves anatomical detail.

        Pass 1: heavy Taubin to crush sensor ripple on the flat sole/arch.
        Pass 2: light Taubin overall to even high-frequency artifacts left
                from decimation/repair.

        Curvature-adaptive in practice: Taubin is intrinsically
        volume-preserving, so high-curvature toes contract less than flats
        per iteration. Splitting the smoothing into two passes lets us run
        more total iterations without compounding detail loss.
        """
        if len(mesh.triangles) < 50:
            return mesh
        mesh = mesh.filter_smooth_taubin(number_of_iterations=base_iter)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=light_iter)
        mesh.compute_vertex_normals()
        return mesh

    # ====================================================================
    # STAGE E — Solidify (extrude to flat base = watertight)
    # ====================================================================

    def _extrude_to_solid(
        self,
        surface: o3d.geometry.TriangleMesh,
        base_offset: float = 0.005,
    ) -> o3d.geometry.TriangleMesh:
        """
        Convert an open sole surface into a closed (watertight) solid by
        extruding to a flat base plane.

        Handles the realistic case of multiple boundary loops:
          - One *outer* loop (the foot silhouette)
          - Possibly *inner* loops (internal Poisson holes left after density
            clipping that fill_holes couldn't close)
        Each loop gets its own sidewall to the base plane. The outermost loop
        also gets a fan-triangulated base cap. Inner loops do NOT — they
        become through-holes in the base, but the *solid* stays watertight
        because each loop is properly closed by sidewalls + ring.

        Better strategy: drop interior loops by filling them at the surface
        level first (we already tried fill_holes), then extrude only the
        outer loop.
        """
        verts = np.asarray(surface.vertices)
        tris = np.asarray(surface.triangles)
        if len(verts) < 20 or len(tris) < 20:
            return surface

        # ---- Find all boundary edges
        edge_count: dict = {}
        for t in tris:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if len(boundary_edges) < 6:
            return surface  # already closed

        # ---- Walk the boundary edges into ordered closed loops
        adj: dict = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        loops = []
        used = set()
        for start in list(adj.keys()):
            if start in used:
                continue
            loop = [start]
            used.add(start)
            prev = None
            current = start
            while True:
                neighbors = [n for n in adj[current] if n != prev]
                if not neighbors:
                    break
                nxt = None
                for n in neighbors:
                    if n == start and len(loop) > 2:
                        nxt = n
                        break
                    if n not in used:
                        nxt = n
                        break
                if nxt is None:
                    break
                if nxt == start:
                    break  # loop closed
                used.add(nxt)
                loop.append(nxt)
                prev = current
                current = nxt
            if len(loop) >= 3:
                loops.append(loop)

        if not loops:
            return surface

        # ---- Identify the OUTER loop = largest XY bounding box area
        def loop_area(loop):
            p = verts[loop]
            return (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
        loops.sort(key=loop_area, reverse=True)
        outer_loop = loops[0]
        inner_loops = loops[1:]

        z_base = float(verts[:, 2].max()) + base_offset

        new_verts = list(verts)
        new_tris = list(tris)

        def extrude_loop(loop, cap: bool):
            """Extrude one closed boundary loop to z_base."""
            base_index = {}
            for v in loop:
                idx = len(new_verts)
                base_index[v] = idx
                p = verts[v].copy()
                p[2] = z_base
                new_verts.append(p)

            n = len(loop)
            # Sidewall
            for k in range(n):
                a = loop[k]
                b = loop[(k + 1) % n]
                a_b = base_index[a]
                b_b = base_index[b]
                new_tris.append([a, b_b, b])
                new_tris.append([a, a_b, b_b])

            if not cap:
                return

            # Fan base cap from a real vertex (no synthetic apex).
            # Angular sort to ensure consistent winding.
            base_pts = np.array([new_verts[base_index[v]] for v in loop])
            cxy = base_pts[:, :2].mean(axis=0)
            rel = base_pts[:, :2] - cxy
            angs = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort(angs)
            ring = [base_index[loop[i]] for i in order]
            anchor = ring[0]
            for k in range(1, len(ring) - 1):
                # Base normal must face +z (downward, away from foot).
                # The original surface faces -z (camera looks up).
                new_tris.append([anchor, ring[k + 1], ring[k]])

        # Outer loop: full extrude + base cap
        extrude_loop(outer_loop, cap=True)
        # Inner loops (interior holes): seal them with sidewalls + their own
        # cap so the volume stays closed. Each interior hole becomes a small
        # cylindrical indent on the base — minor cosmetic, but watertight.
        for il in inner_loops:
            if len(il) >= 3:
                extrude_loop(il, cap=True)

        solid = o3d.geometry.TriangleMesh()
        solid.vertices = o3d.utility.Vector3dVector(
            np.array(new_verts, dtype=np.float64))
        solid.triangles = o3d.utility.Vector3iVector(
            np.array(new_tris, dtype=np.int32))
        solid.compute_vertex_normals()
        solid.compute_triangle_normals()
        return solid

    # ====================================================================
    # STAGE F — Manifold repair, decimation, cleanup
    # ====================================================================

    def _close_interior_holes(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """
        Close any interior boundary loops with planar caps.

        After Poisson + density crop + silhouette clip, the surface can have
        small interior holes (Poisson craters). Open3D's built-in fill_holes
        is size-limited. This pass finds every closed boundary loop EXCEPT
        the largest (= outer foot silhouette) and triangulates each as a
        flat cap at the loop's mean depth. Result: a clean surface with
        ONE boundary loop (the foot outline), ready for extrusion.
        """
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        if len(tris) < 100:
            return mesh

        # Find boundary edges
        edge_count: dict = {}
        for t in tris:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if len(boundary_edges) < 6:
            return mesh

        # Walk into loops
        adj: dict = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        loops = []
        visited = set()
        for start in list(adj.keys()):
            if start in visited:
                continue
            loop = [start]
            visited.add(start)
            prev, cur = None, start
            while True:
                nbs = [n for n in adj[cur] if n != prev]
                nxt = None
                for n in nbs:
                    if n == start and len(loop) > 2:
                        nxt = n
                        break
                    if n not in visited:
                        nxt = n
                        break
                if nxt is None or nxt == start:
                    break
                visited.add(nxt)
                loop.append(nxt)
                prev, cur = cur, nxt
            if len(loop) >= 3:
                loops.append(loop)

        if len(loops) <= 1:
            return mesh

        # Largest XY bbox = outer (foot silhouette). Keep that one open.
        def bbox_area(loop):
            p = verts[loop]
            return (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
        loops.sort(key=bbox_area, reverse=True)
        outer = loops[0]
        interior = loops[1:]

        new_tris = list(tris)
        for loop in interior:
            if len(loop) < 3:
                continue
            # Fan cap from a real loop vertex (no synthetic apex).
            # Loop winding determines normal direction; we'll let the final
            # repair pass fix orientation if needed.
            anchor = loop[0]
            for k in range(1, len(loop) - 1):
                new_tris.append([anchor, loop[k], loop[k + 1]])

        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(verts.copy())
        out.triangles = o3d.utility.Vector3iVector(
            np.array(new_tris, dtype=np.int32))
        out.remove_duplicated_triangles()
        out.remove_degenerate_triangles()
        out.compute_vertex_normals()
        out.compute_triangle_normals()
        return out

    def _final_repair(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """Post-extrusion manifold cleanup (no decimation here)."""
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        # Iterate non-manifold-vertex removal until stable
        for _ in range(5):
            if mesh.is_vertex_manifold():
                break
            try:
                nm = np.asarray(mesh.get_non_manifold_vertices())
            except Exception:
                break
            if len(nm) == 0:
                break
            bad = np.zeros(len(mesh.vertices), dtype=bool)
            bad[nm] = True
            mesh.remove_vertices_by_mask(bad)
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def _repair_and_decimate(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 60_000,
    ) -> o3d.geometry.TriangleMesh:
        # First pass cleanup
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles
            )

        # Aggressive post-decimation cleanup. Decimation can introduce new
        # non-manifold edges/vertices; do another pass.
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        # If still non-vertex-manifold, iteratively drop offending vertices.
        # Each removal can expose a new non-manifold vertex; loop until clean
        # or no progress.
        for _ in range(5):
            if mesh.is_vertex_manifold():
                break
            try:
                nm = np.asarray(mesh.get_non_manifold_vertices())
            except Exception:
                break
            if len(nm) == 0:
                break
            bad = np.zeros(len(mesh.vertices), dtype=bool)
            bad[nm] = True
            mesh.remove_vertices_by_mask(bad)
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def _keep_largest_component(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        try:
            clusters, n_tri, _ = mesh.cluster_connected_triangles()
        except Exception:
            return mesh
        n_tri = np.asarray(n_tri)
        if len(n_tri) <= 1:
            return mesh
        largest = int(n_tri.argmax())
        remove = np.asarray(clusters) != largest
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
        return mesh

    # ====================================================================
    # STAGE G — Standardize for CAD export
    # ====================================================================

    def standardize_for_export(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """Scale to mm, center XY, base at Z=0, foot points +Z."""
        if len(mesh.vertices) == 0:
            return mesh
        mesh.scale(1000.0, center=mesh.get_center())
        bbox = mesh.get_axis_aligned_bounding_box()
        c = bbox.get_center()
        mesh.translate((-c[0], -c[1], 0.0))
        verts = np.asarray(mesh.vertices)
        # Mesh currently has sole at z_min (negative-ish) and base at z_max.
        # Flip to put base at z=0 and sole pointing +z (the foot rises up).
        R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        mesh.rotate(R, center=(0, 0, 0))
        verts = np.asarray(mesh.vertices)
        mesh.translate((0.0, 0.0, -verts[:, 2].min()))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    # ====================================================================
    # PUBLIC — single-frame pipeline
    # ====================================================================

    def reconstruct_from_depth(
        self,
        depth_isolated: np.ndarray,
        output_path: Optional[str] = None,
        target_triangles: int = 60_000,
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = None,   # kept for caller signature compat
    ) -> o3d.geometry.TriangleMesh:
        """
        End-to-end single-frame reconstruction.

        Args:
            depth_isolated:  refined+upsampled foot depth map (NaN = invalid)
            output_path:     optional .stl path
            target_triangles: final triangle budget
            mask:            optional uint8 foot silhouette (same shape as
                             depth_isolated). When provided, the mesh boundary
                             is clipped to this silhouette — strongly improves
                             edge accuracy.
            method:          ignored; kept so the existing pipeline call works
        """
        print("  [1/8] Back-projecting depth with analytical normals")
        pcd = self._depth_to_oriented_pcd(depth_isolated)
        if len(pcd.points) < 200:
            raise ValueError(
                f"Reconstruction input too sparse: {len(pcd.points)} points")

        print(f"  [2/8] Poisson reconstruction (depth=9, density crop)")
        mesh = self._poisson_with_density_crop(pcd, depth=9,
                                               density_percentile=6.0)
        if len(mesh.triangles) < 100:
            raise ValueError("Poisson produced an empty mesh")

        print("  [3/8] Clip mesh boundary to foot silhouette")
        mesh = self._clip_to_silhouette(mesh, mask, depth_isolated.shape)

        print("  [4/8] Keep largest connected component")
        mesh = self._keep_largest_component(mesh)

        # Fill internal holes in the surface BEFORE extrusion — only the
        # outer foot silhouette should remain as a boundary loop. Two-pass:
        # Open3D's built-in fill_holes (size-limited), then planar caps for
        # any remaining interior loops, leaving only the outer boundary.
        try:
            mesh = mesh.fill_holes(hole_size=0.05)
        except Exception:
            pass
        mesh = self._close_interior_holes(mesh)

        print("  [5/8] Curvature-adaptive smoothing")
        mesh = self._adaptive_smooth(mesh, base_iter=18, light_iter=6)

        # Decimate the OPEN SURFACE first (not the solid). If we decimated
        # after extrusion, the simplifier would remove boundary triangles
        # and leave the solid full of small holes / unclosed edges. Doing it
        # here means: surface → decimate → extrude (sidewalls + cap are
        # small, cheap, never touched by decimation).
        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles
            )
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        print("  [6/8] Extrude to flat base — closed solid")
        # base_offset 1cm — large enough to never self-intersect after the
        # surface curvature/normals push the surface around.
        mesh = self._extrude_to_solid(mesh, base_offset=0.010)

        print("  [7/8] Manifold repair")
        mesh = self._final_repair(mesh)

        print("  [8/8] Standardize to mm, CAD-ready orientation")
        mesh = self.standardize_for_export(mesh)

        m = self.get_mesh_metrics(mesh)
        print(f"  Done: {m}")

        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)
        return mesh

    # Multi-frame entry kept for caller signature compatibility — degrades to
    # a temporal median, then runs the single-frame pipeline.
    def fuse_stationary_frames(
        self,
        depth_frames: List[np.ndarray],
        output_path: Optional[str] = None,
        target_triangles: int = 60_000,
        mask: Optional[np.ndarray] = None,
    ) -> o3d.geometry.TriangleMesh:
        print(f"\n--- Temporal median over {len(depth_frames)} frames ---")
        stack = np.stack(depth_frames, axis=0).astype(np.float32)
        stack[stack <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fused = np.nanmedian(stack, axis=0)
        fused[np.isnan(fused)] = 0.0
        fused[fused <= 0] = np.nan
        return self.reconstruct_from_depth(
            depth_isolated=fused, output_path=output_path,
            target_triangles=target_triangles, mask=mask,
        )

    # ====================================================================
    # Metrics & measurements (unchanged API)
    # ====================================================================

    def get_mesh_metrics(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return {}
        bounds = pts.max(axis=0) - pts.min(axis=0)
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "is_watertight": mesh.is_watertight(),
            "bounds_xyz_mm": [round(float(b), 1) for b in bounds],
            "foot_length_mm": float(max(bounds[0], bounds[1])),
        }

    def measure_foot(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return {"length_mm": 0, "width_mm": 0, "height_mm": 0,
                    "eu_size_approx": 0, "vertices": 0, "triangles": 0}
        extents = pts.max(axis=0) - pts.min(axis=0)
        s = np.sort(extents)[::-1]
        length, width, height = float(s[0]), float(s[1]), float(s[2])
        return {
            "length_mm": round(length, 1),
            "width_mm": round(width, 1),
            "height_mm": round(height, 1),
            "eu_size_approx": round(length / 6.67),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        }