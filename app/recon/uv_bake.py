"""
backend/app/recon/uv_bake.py
UV unwrap (xatlas) + texture bake from RGB frames.

If xatlas not installed: falls back to planar projection UV.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

try:
    import xatlas
    _HAS_XATLAS = True
except ImportError:
    _HAS_XATLAS = False


def unwrap_and_bake(mesh: o3d.geometry.TriangleMesh,
                    raw_dir: Path,
                    tex_size: int = 2048
                    ) -> tuple[o3d.geometry.TriangleMesh, Optional[Path]]:
    """
    Returns (mesh_with_uvs, path_to_diffuse_png).
    raw_dir: directory containing rgb_*.png frames.
    """
    out_png = raw_dir.parent / "foot_diffuse.png"

    if _HAS_XATLAS:
        mesh = _xatlas_unwrap(mesh)
    else:
        mesh = _planar_uv(mesh)

    _bake_texture(mesh, raw_dir, out_png, size=tex_size)
    return mesh, out_png


# ─────────────── xatlas parametrize ───────────────

def _xatlas_unwrap(mesh: o3d.geometry.TriangleMesh
                   ) -> o3d.geometry.TriangleMesh:
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.triangles, dtype=np.uint32)

    vmapping, indices, uvs = xatlas.parametrize(v, f)

    # rebuild mesh with remapped vertices
    new_v = v[vmapping]
    new_f = indices.astype(np.int32)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_v.astype(np.float64))
    out.triangles = o3d.utility.Vector3iVector(new_f)

    # per-triangle UVs (flat: 3 * nT rows, 2 cols)
    tri_uvs = uvs[new_f].reshape(-1, 2)
    out.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs.astype(np.float64))

    # carry over vertex colors if present
    if mesh.has_vertex_colors():
        old_colors = np.asarray(mesh.vertex_colors)
        out.vertex_colors = o3d.utility.Vector3dVector(old_colors[vmapping])

    out.compute_vertex_normals()
    return out


# ─────────────── planar fallback ───────────────

def _planar_uv(mesh: o3d.geometry.TriangleMesh
               ) -> o3d.geometry.TriangleMesh:
    """Simple XY planar projection as UV."""
    v = np.asarray(mesh.vertices)
    xy = v[:, :2].copy()
    lo = xy.min(0)
    rng = xy.ptp(0)
    rng[rng < 1e-8] = 1.0
    uv = (xy - lo) / rng                # [N, 2] in [0,1]

    tris = np.asarray(mesh.triangles)
    tri_uvs = uv[tris].reshape(-1, 2)   # [3*nT, 2]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs.astype(np.float64))
    return mesh


# ─────────────── texture baking ───────────────

def _bake_texture(mesh: o3d.geometry.TriangleMesh,
                  raw_dir: Path,
                  out_png: Path,
                  size: int = 2048) -> None:
    """
    Rasterize triangles into UV space, fill with vertex colors or
    averaged face color from nearest RGB frame.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)

    has_uvs = mesh.has_triangle_uvs() and len(mesh.triangle_uvs) > 0
    if not has_uvs:
        cv2.imwrite(str(out_png), img)
        return

    uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)   # [nT, 3, 2]
    tris = np.asarray(mesh.triangles)
    has_vcol = mesh.has_vertex_colors()
    vcol = np.asarray(mesh.vertex_colors) if has_vcol else None

    for i in range(len(tris)):
        a, b, c = tris[i]
        if has_vcol:
            col = ((vcol[a] + vcol[b] + vcol[c]) / 3.0 * 255).astype(np.uint8)
            col_bgr = col[::-1].tolist()
        else:
            col_bgr = [200, 200, 200]

        pts_uv = (uvs[i] * size).astype(np.int32)
        pts_uv[:, 1] = size - 1 - pts_uv[:, 1]   # flip Y for image coords
        cv2.fillPoly(img, [pts_uv], col_bgr)

    cv2.imwrite(str(out_png), img)
