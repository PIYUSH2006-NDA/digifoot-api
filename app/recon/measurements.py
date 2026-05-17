"""
backend/app/recon/measurements.py
Extract clinical foot measurements from a cleaned mesh.

Returns dict with mm values:
  length_mm, ball_width_mm, heel_width_mm, arch_height_mm, instep_girth_mm
"""
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


def extract_measurements(mesh: o3d.geometry.TriangleMesh) -> dict:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) < 100:
        return _empty()

    # canonical align: PCA, long axis → X, short → Z
    v = _canonical_align(v)

    # detect scale: if bbox length < 5 → assume meters, convert to mm
    L = v[:, 0].ptp()
    if L < 5.0:
        v *= 1000.0
        L *= 1000.0

    # ── length: heel to longest toe along X ──
    length = float(L)

    # ── ball width: slice at 72% from heel ──
    ball_w = _slice_width(v, frac=0.72, axis=0, measure_axis=1)

    # ── heel width: slice at 15% from heel ──
    heel_w = _slice_width(v, frac=0.15, axis=0, measure_axis=1)

    # ── arch height: midfoot Z range at 50% ──
    arch_h = _slice_range(v, frac=0.50, axis=0, measure_axis=2, half_w=10.0)

    # ── instep girth: convex hull perimeter of cross-section at 55% ──
    instep = _hull_perimeter_at(v, frac=0.55, axis=0, half_w=5.0)

    return {
        "length_mm":       round(length, 2),
        "ball_width_mm":   round(ball_w, 2),
        "heel_width_mm":   round(heel_w, 2),
        "arch_height_mm":  round(arch_h, 2),
        "instep_girth_mm": round(instep, 2),
    }


# ─────────────── internals ───────────────

def _empty() -> dict:
    return {k: 0.0 for k in
            ["length_mm", "ball_width_mm", "heel_width_mm",
             "arch_height_mm", "instep_girth_mm"]}


def _canonical_align(v: np.ndarray) -> np.ndarray:
    c = v.mean(0)
    v = v - c
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    Q = eigvecs[:, order]
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    v = v @ Q
    # ensure sole is at low Z
    upper = (v[:, 2] > v[:, 2].mean()).sum()
    lower = (v[:, 2] < v[:, 2].mean()).sum()
    if upper > lower:
        v[:, 2] *= -1
    # ensure heel at low X (heel has fewer vertices = narrower cross-section)
    x_mid = v[:, 0].mean()
    front = v[v[:, 0] > x_mid]
    back = v[v[:, 0] < x_mid]
    if len(back) > 0 and len(front) > 0:
        front_w = front[:, 1].ptp()
        back_w = back[:, 1].ptp()
        if front_w < back_w:
            v[:, 0] *= -1
    return v


def _slice_width(v: np.ndarray, frac: float, axis: int,
                 measure_axis: int, half_w: float = 5.0) -> float:
    lo = v[:, axis].min()
    rng = v[:, axis].ptp()
    center = lo + frac * rng
    mask = np.abs(v[:, axis] - center) < half_w
    if mask.sum() < 5:
        return 0.0
    return float(v[mask, measure_axis].ptp())


def _slice_range(v: np.ndarray, frac: float, axis: int,
                 measure_axis: int, half_w: float = 10.0) -> float:
    lo = v[:, axis].min()
    center = lo + frac * v[:, axis].ptp()
    mask = np.abs(v[:, axis] - center) < half_w
    if mask.sum() < 5:
        return 0.0
    return float(v[mask, measure_axis].ptp())


def _hull_perimeter_at(v: np.ndarray, frac: float, axis: int,
                       half_w: float = 5.0) -> float:
    lo = v[:, axis].min()
    center = lo + frac * v[:, axis].ptp()
    mask = np.abs(v[:, axis] - center) < half_w
    pts = v[mask]
    if len(pts) < 5:
        return 0.0
    # project to 2D: keep the two axes that are NOT `axis`
    axes = [i for i in range(3) if i != axis]
    pts2d = pts[:, axes]
    try:
        hull = ConvexHull(pts2d)
        verts = pts2d[hull.vertices]
        verts_closed = np.vstack([verts, verts[0:1]])
        diffs = np.diff(verts_closed, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    except Exception:
        return 0.0
