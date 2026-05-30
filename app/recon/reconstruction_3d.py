# reconstruction_3d.py  (v9 — Template-Fit Foot Reconstruction)
#
# Single-frame TrueDepth scan → measure foot proportions → warp canonical
# foot template to match → individualized foot mesh.
#
# Pipeline (FootReconstructor.reconstruct_from_depth):
#   1. Extract scan anchors (length-axis, outline, landmarks, sole-Z)
#   2. Load + cache template (right STL + auto-mirrored left)
#   3. TPS warp of template's 6 landmark points to scan landmarks
#   4. Light blend of template sole vertices toward scan-measured Z
#   5. Standardize to mm, CAD-ready orientation, STL export
#
# Falls back to a basic Poisson path if the template is missing or anchor
# extraction fails — so the system stays alive.

import numpy as np
import open3d as o3d
import cv2
import warnings
from pathlib import Path
from typing import List, Optional


# ───────────────────────────────────────────────────────────────────────
#  Template store — loaded once per process
# ───────────────────────────────────────────────────────────────────────

class FootTemplateStore:
    """Caches canonical right template + auto-mirrored left."""

    _instance: Optional["FootTemplateStore"] = None

    def __init__(self):
        self.right: Optional[o3d.geometry.TriangleMesh] = None
        self.left: Optional[o3d.geometry.TriangleMesh] = None
        self._load()

    @classmethod
    def get(cls) -> "FootTemplateStore":
        if cls._instance is None:
            cls._instance = FootTemplateStore()
        return cls._instance

    def _candidate_paths(self):
        here = Path(__file__).resolve().parent
        return [
            Path("weights/templates/foot_template_right.stl"),
            Path("weights/foot_template_right.stl"),
            here / "templates" / "foot_template_right.stl",
            here.parent / "weights" / "templates" / "foot_template_right.stl",
            here.parent / "weights" / "foot_template_right.stl",
            here.parent.parent / "weights" / "foot_template_right.stl",
            Path("/app/weights/templates/foot_template_right.stl"),
            Path("/app/weights/foot_template_right.stl"),
        ]

    def _load(self):
        for p in self._candidate_paths():
            if p.exists():
                try:
                    m = o3d.io.read_triangle_mesh(str(p))
                    if len(m.triangles) > 0:
                        self.right = self._canonicalize(m)
                        self.left = self._mirror_x(self.right)
                        print(f"[template] loaded {p}  verts={len(self.right.vertices)} tris={len(self.right.triangles)}")
                        return
                except Exception as e:
                    print(f"[template] failed to load {p}: {e}")
                    continue
        print("[template] not found — pipeline will use legacy reconstruction fallback")

    @staticmethod
    def _canonicalize(m: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        v = np.asarray(m.vertices)
        c = v.mean(axis=0)
        v = v - np.array([c[0], c[1], 0])
        if v[:, 2].mean() > 0:
            v = v * np.array([1, 1, -1])
            tris = np.asarray(m.triangles)[:, [0, 2, 1]]
            m.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
        m.vertices = o3d.utility.Vector3dVector(v)
        m.compute_vertex_normals()
        return m

    @staticmethod
    def _mirror_x(right: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        left = o3d.geometry.TriangleMesh(right)
        v = np.asarray(left.vertices).copy()
        v[:, 0] = -v[:, 0]
        t = np.asarray(left.triangles).copy()[:, [0, 2, 1]]
        left.vertices = o3d.utility.Vector3dVector(v)
        left.triangles = o3d.utility.Vector3iVector(t.astype(np.int32))
        left.compute_vertex_normals()
        return left

    def get_for_side(self, side: str) -> Optional[o3d.geometry.TriangleMesh]:
        s = (side or "right").lower()
        src = self.left if s.startswith("l") else self.right
        if src is None:
            return None
        m = o3d.geometry.TriangleMesh(src)
        m.vertices = o3d.utility.Vector3dVector(np.asarray(src.vertices).copy())
        m.triangles = o3d.utility.Vector3iVector(np.asarray(src.triangles).copy())
        m.compute_vertex_normals()
        return m


# ───────────────────────────────────────────────────────────────────────
#  Scan anchor extraction
# ───────────────────────────────────────────────────────────────────────

def _extract_scan_anchors(
    depth_isolated: np.ndarray,
    mask: Optional[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
) -> Optional[dict]:
    """Extract length axis, outline, landmarks, sole Z. None if too sparse."""
    h, w = depth_isolated.shape
    valid = ~np.isnan(depth_isolated) & (depth_isolated > 0)
    if mask is not None and mask.shape == depth_isolated.shape:
        valid &= (mask > 0)
    if valid.sum() < 500:
        return None

    i_grid, j_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_isolated.astype(np.float32)
    xs = (j_grid - cx) * z / fx
    ys = (i_grid - cy) * z / fy
    pts3 = np.stack([xs[valid], ys[valid], z[valid]], axis=1)
    pts2 = pts3[:, :2]

    # PCA for length axis
    centroid2 = pts2.mean(axis=0)
    centered = pts2 - centroid2
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    length_axis = eigvecs[:, order[0]]
    width_axis = eigvecs[:, order[1]]

    s = centered @ length_axis
    t = centered @ width_axis
    s_lo, s_hi = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))
    t_lo, t_hi = float(np.quantile(t, 0.01)), float(np.quantile(t, 0.99))
    length_m = s_hi - s_lo
    width_m = t_hi - t_lo

    # Heel = narrower end, toes = wider end
    band = length_m * 0.10
    end_lo_t = t[(s >= s_lo) & (s <= s_lo + band)]
    end_hi_t = t[(s >= s_hi - band) & (s <= s_hi)]
    lo_w = float(end_lo_t.max() - end_lo_t.min()) if len(end_lo_t) else 0
    hi_w = float(end_hi_t.max() - end_hi_t.min()) if len(end_hi_t) else 0
    if lo_w > hi_w:
        # Heel was at high end — flip convention
        length_axis = -length_axis
        s = -s
        s_lo, s_hi = -s_hi, -s_lo

    front_half = s > (s_lo + s_hi) / 2
    ball_width_m = float(t[front_half].max() - t[front_half].min()) if front_half.any() else width_m
    sole_z_mean = float(pts3[:, 2].mean())

    return {
        "pts3": pts3, "pts2": pts2,
        "centroid2": centroid2,
        "length_axis": length_axis, "width_axis": width_axis,
        "length_m": length_m, "width_m": width_m,
        "ball_width_m": ball_width_m,
        "s_lo": s_lo, "s_hi": s_hi,
        "t_lo": t_lo, "t_hi": t_hi,
        "sole_z_mean": sole_z_mean,
    }


# ───────────────────────────────────────────────────────────────────────
#  Thin-plate-spline warp (3D, NumPy-only)
# ───────────────────────────────────────────────────────────────────────

def _tps_solve(src_pts: np.ndarray, dst_pts: np.ndarray,
               regularization: float = 1e-6):
    k, d = src_pts.shape
    diff = src_pts[:, None, :] - src_pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
    # Biharmonic kernel for 3D TPS: -r
    K = -dist + regularization * np.eye(k)
    P = np.concatenate([np.ones((k, 1)), src_pts], axis=1)
    L = np.zeros((k + d + 1, k + d + 1))
    L[:k, :k] = K
    L[:k, k:] = P
    L[k:, :k] = P.T
    Y = np.zeros((k + d + 1, d))
    Y[:k] = dst_pts
    sol = np.linalg.solve(L, Y)
    return sol[:k], sol[k:], src_pts


def _tps_apply(query_pts: np.ndarray, W: np.ndarray, A: np.ndarray,
               src_pts: np.ndarray) -> np.ndarray:
    diff = query_pts[:, None, :] - src_pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
    U = -dist
    P = np.concatenate([np.ones((query_pts.shape[0], 1)), query_pts], axis=1)
    return U @ W + P @ A


# ───────────────────────────────────────────────────────────────────────
#  Template fitting
# ───────────────────────────────────────────────────────────────────────

def _fit_template(
    template: o3d.geometry.TriangleMesh,
    anchors: dict,
    side: str,
) -> o3d.geometry.TriangleMesh:
    """Warp template (mm, template frame) to scan anchors (m, camera frame)."""
    tv = np.asarray(template.vertices).astype(np.float64) / 1000.0  # mm → m
    t_y = tv[:, 1]; t_x = tv[:, 0]; t_z = tv[:, 2]
    t_y_min, t_y_max = float(t_y.min()), float(t_y.max())
    t_length = t_y_max - t_y_min

    # Template control points (in meters, template frame)
    heel_t = np.array([0.0, t_y_min, t_z[t_y < t_y_min + 0.1 * t_length].mean()])
    toe_t  = np.array([0.0, t_y_max, t_z[t_y > t_y_max - 0.1 * t_length].mean()])
    band = t_length * 0.05
    ball_y = t_y_min + 0.7 * t_length
    bm = (t_y > ball_y - band) & (t_y < ball_y + band)
    if bm.sum() > 5:
        ball_L_t = np.array([float(t_x[bm].min()), ball_y, float(t_z[bm].mean())])
        ball_R_t = np.array([float(t_x[bm].max()), ball_y, float(t_z[bm].mean())])
    else:
        ball_L_t = np.array([-0.04, ball_y, 0.0])
        ball_R_t = np.array([0.04, ball_y, 0.0])
    mid_y = t_y_min + 0.4 * t_length
    mm = (t_y > mid_y - band) & (t_y < mid_y + band)
    if mm.sum() > 5:
        mid_L_t = np.array([float(t_x[mm].min()), mid_y, float(t_z[mm].mean())])
        mid_R_t = np.array([float(t_x[mm].max()), mid_y, float(t_z[mm].mean())])
    else:
        mid_L_t = np.array([-0.03, mid_y, 0.0])
        mid_R_t = np.array([0.03, mid_y, 0.0])

    src_3d = np.stack([heel_t, toe_t, ball_L_t, ball_R_t, mid_L_t, mid_R_t])

    # Top-of-foot anchors to preserve template height during TPS warp.
    # Without these, all 6 sole anchors are near z=0 → TPS pulls the whole
    # template down to the sole plane, collapsing the 49mm template height
    # to ~7mm in output. The top anchors hold the dorsum (top of foot) up
    # at the template's natural height, scaled proportionally to scan
    # length (taller people / longer feet generally have taller insteps).
    top_z_template = float(t_z.max())   # ~+27mm in mm = 0.027 m
    # Heel-top (back of ankle): just above the heel landmark
    heel_top_t = np.array([0.0, t_y_min + 0.05 * t_length, top_z_template * 0.6])
    # Instep peak: roughly between ball and midfoot, where dorsum is highest
    instep_y = t_y_min + 0.55 * t_length
    instep_t = np.array([0.0, instep_y, top_z_template])
    # Toe-top: above the toe landmark, lower than instep
    toe_top_t = np.array([0.0, t_y_max - 0.02 * t_length, top_z_template * 0.55])

    src_3d = np.concatenate([src_3d, np.stack([heel_top_t, instep_t, toe_top_t])])

    # Scan-side counterparts in camera-frame meters
    centroid2 = anchors["centroid2"]
    L = anchors["length_axis"]; Wv = anchors["width_axis"]
    s_lo, s_hi = anchors["s_lo"], anchors["s_hi"]
    t_lo_s, t_hi_s = anchors["t_lo"], anchors["t_hi"]
    sole_z = anchors["sole_z_mean"]
    length_s = anchors["length_m"]

    def world_from_st(s_val, t_val, z_val):
        p2 = centroid2 + s_val * L + t_val * Wv
        return np.array([p2[0], p2[1], z_val])

    heel_s = world_from_st(s_lo, 0.0, sole_z)
    toe_s  = world_from_st(s_hi, 0.0, sole_z)
    ball_at = s_lo + 0.7 * length_s
    ball_L_s = world_from_st(ball_at, t_lo_s, sole_z)
    ball_R_s = world_from_st(ball_at, t_hi_s, sole_z)
    mid_at = s_lo + 0.4 * length_s
    mid_L_s = world_from_st(mid_at, t_lo_s * 0.6, sole_z)
    mid_R_s = world_from_st(mid_at, t_hi_s * 0.6, sole_z)
    dst_3d = np.stack([heel_s, toe_s, ball_L_s, ball_R_s, mid_L_s, mid_R_s])

    # Scan-side top anchors: place them at the heights derived from the
    # TEMPLATE (in meters), but in scan world space (camera frame). We
    # scale template-height anchors by the ratio of scan length to template
    # length so taller scans get proportionally taller feet.
    template_length_m = t_length
    height_scale = length_s / template_length_m
    # Camera-frame Z grows AWAY from camera. Sole is at sole_z (closest);
    # top of foot is at sole_z - height (FARTHER from sole, but since the
    # camera is BELOW pointing up, top is farther — smaller z? Actually,
    # camera looks UP at the sole. Sole = closest pixel = smallest z.
    # Top of foot is HIGHER above the camera = LARGER z. So:
    # top_z = sole_z + (template_top_z in scan-scaled meters)
    def world_top(s_val, top_template_mm):
        p2 = centroid2 + s_val * L
        # Template top is in template Z (mm); scan top height = scaled
        top_height_m = (top_template_mm / 1000.0) * height_scale
        return np.array([p2[0], p2[1], sole_z + top_height_m])

    heel_top_s = world_top(s_lo + 0.05 * length_s, top_z_template * 1000 * 0.6)
    instep_s = world_top(s_lo + 0.55 * length_s, top_z_template * 1000)
    toe_top_s = world_top(s_hi - 0.02 * length_s, top_z_template * 1000 * 0.55)

    dst_3d = np.concatenate([dst_3d, np.stack([heel_top_s, instep_s, toe_top_s])])

    W, A, src = _tps_solve(src_3d, dst_3d)
    warped = _tps_apply(tv, W, A, src)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(warped)
    out.triangles = o3d.utility.Vector3iVector(
        np.asarray(template.triangles).astype(np.int32))
    out.compute_vertex_normals()
    out.compute_triangle_normals()
    return out


# ───────────────────────────────────────────────────────────────────────
#  Sole detail blend
# ───────────────────────────────────────────────────────────────────────

def _blend_sole_detail(
    mesh: o3d.geometry.TriangleMesh,
    depth_isolated: np.ndarray,
    mask: Optional[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    weight: float = 0.35,
) -> o3d.geometry.TriangleMesh:
    """Pull template sole verts toward scan-measured depth (Z). Light weight."""
    v = np.asarray(mesh.vertices).copy()
    mesh.compute_vertex_normals()
    n = np.asarray(mesh.vertex_normals)
    sole_mask = n[:, 2] < -0.3
    if sole_mask.sum() < 10:
        return mesh

    sole_v = v[sole_mask]
    z = sole_v[:, 2]
    valid_z = z > 0
    if valid_z.sum() < 10:
        return mesh
    j = (sole_v[valid_z, 0] * fx / z[valid_z]) + cx
    i = (sole_v[valid_z, 1] * fy / z[valid_z]) + cy
    h, w = depth_isolated.shape
    ji = (j.astype(int).clip(0, w - 1), i.astype(int).clip(0, h - 1))
    measured_z = depth_isolated[ji[1], ji[0]]
    in_mask = ~np.isnan(measured_z) & (measured_z > 0)
    if mask is not None:
        in_mask &= (mask[ji[1], ji[0]] > 0)
    if in_mask.sum() < 5:
        return mesh

    sub_idx = np.where(sole_mask)[0][valid_z][in_mask]
    measured_in = measured_z[in_mask]
    old_z = v[sub_idx, 2]
    v[sub_idx, 2] = (1.0 - weight) * old_z + weight * measured_in

    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# ───────────────────────────────────────────────────────────────────────
#  FootReconstructor
# ───────────────────────────────────────────────────────────────────────

class FootReconstructor:
    """Template-fit primary; basic Poisson fallback."""

    def __init__(self, preprocessor):
        self.prep = preprocessor
        self._templates = FootTemplateStore.get()

    def reconstruct_from_depth(
        self,
        depth_isolated: np.ndarray,
        output_path: Optional[str] = None,
        target_triangles: int = 60_000,
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        side: str = "right",
    ) -> o3d.geometry.TriangleMesh:
        print("  [1/5] Extracting scan anchors")
        anchors = _extract_scan_anchors(
            depth_isolated, mask,
            self.prep.fx, self.prep.fy, self.prep.cx, self.prep.cy)
        if anchors is None:
            print("  ⚠ anchor extraction failed — legacy fallback")
            return self._legacy_reconstruct(
                depth_isolated, mask, target_triangles, output_path)

        L_mm = anchors["length_m"] * 1000
        W_mm = anchors["width_m"] * 1000
        BW_mm = anchors["ball_width_m"] * 1000
        Z_mm = anchors["sole_z_mean"] * 1000
        print(f"      length={L_mm:.1f}mm  width={W_mm:.1f}mm  "
              f"ball_width={BW_mm:.1f}mm  sole_z={Z_mm:.1f}mm")

        template = self._templates.get_for_side(side)
        if template is None:
            print("  ⚠ template not found — legacy fallback")
            return self._legacy_reconstruct(
                depth_isolated, mask, target_triangles, output_path)

        if L_mm < 120 or L_mm > 350:
            print(f"  ⚠ scan length {L_mm:.0f}mm outside [120, 350]mm — legacy fallback")
            return self._legacy_reconstruct(
                depth_isolated, mask, target_triangles, output_path)

        print(f"  [2/5] Fitting template ({side}) to scan anchors")
        fitted = _fit_template(template, anchors, side)

        print("  [3/5] Blending scan sole detail into template")
        fitted = _blend_sole_detail(
            fitted, depth_isolated, mask,
            self.prep.fx, self.prep.fy, self.prep.cx, self.prep.cy,
            weight=0.35)

        print("  [4/5] Decimation + manifold repair")
        fitted = self._repair_and_decimate(fitted, target_triangles)

        print("  [5/5] Standardize to mm, CAD-ready orientation")
        fitted = self.standardize_for_export(fitted)

        m = self.get_mesh_metrics(fitted)
        print(f"  Done: {m}")
        if output_path:
            o3d.io.write_triangle_mesh(output_path, fitted)
        return fitted

    def fuse_stationary_frames(
        self,
        depth_frames: List[np.ndarray],
        output_path: Optional[str] = None,
        target_triangles: int = 60_000,
        mask: Optional[np.ndarray] = None,
        side: str = "right",
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
            target_triangles=target_triangles, mask=mask, side=side)

    def standardize_for_export(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        if len(mesh.vertices) == 0:
            return mesh
        v = np.asarray(mesh.vertices).copy()
        scale = v.max(axis=0) - v.min(axis=0)
        if scale.max() < 1.0:
            v = v * 1000.0
        v[:, 0] -= (v[:, 0].max() + v[:, 0].min()) / 2
        v[:, 1] -= (v[:, 1].max() + v[:, 1].min()) / 2
        # Camera-frame z: small = closer (sole), big = farther (top of foot).
        # CAD convention: foot pointing +Z, base at Z=0.
        v[:, 2] = -v[:, 2]
        v[:, 2] -= v[:, 2].min()
        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(v)
        out.triangles = o3d.utility.Vector3iVector(
            np.asarray(mesh.triangles).astype(np.int32))
        out.compute_vertex_normals()
        out.compute_triangle_normals()
        return out

    def _repair_and_decimate(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 60_000,
    ) -> o3d.geometry.TriangleMesh:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
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

    def _legacy_reconstruct(
        self,
        depth_isolated: np.ndarray,
        mask: Optional[np.ndarray],
        target_triangles: int,
        output_path: Optional[str],
    ) -> o3d.geometry.TriangleMesh:
        h, w = depth_isolated.shape
        valid = ~np.isnan(depth_isolated) & (depth_isolated > 0)
        if mask is not None and mask.shape == depth_isolated.shape:
            valid &= mask > 0
        if valid.sum() < 200:
            raise ValueError("Fallback: depth too sparse")
        i_grid, j_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z = depth_isolated.astype(np.float32)
        x = (j_grid - self.prep.cx) * z / self.prep.fx
        y = (i_grid - self.prep.cy) * z / self.prep.fy
        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh, dens = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, scale=1.05, linear_fit=True))
        dens = np.asarray(dens)
        if len(dens):
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06))
        mesh = self._repair_and_decimate(mesh, target_triangles)
        mesh = self.standardize_for_export(mesh)
        if output_path:
            o3d.io.write_triangle_mesh(output_path, mesh)
        return mesh