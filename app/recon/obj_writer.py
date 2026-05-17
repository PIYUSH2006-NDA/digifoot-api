"""
backend/app/recon/obj_writer.py
Writes OBJ + MTL + PNG in same format as the reference Lucas/Rhino file:

  # Rhino\r\n
  mtllib foot.mtl\r\n
  usemtl foot_mat\r\n
  v x y z            (16 decimal places)\r\n
  vt u v             (20 decimal places)\r\n
  vn x y z           (16 decimal places)\r\n
  f a/a/a b/b/b c/c/c  (1-indexed, triangulated)\r\n

All lines \r\n terminated. Vertex/normal 16dp, UV 20dp.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d


def write_obj_rhino(mesh: o3d.geometry.TriangleMesh,
                    tex_png: Optional[Path],
                    out_path: Path,
                    name: str = "foot") -> None:
    out_path = Path(out_path)
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    t = np.asarray(mesh.triangles)

    has_uv = mesh.has_triangle_uvs() and len(mesh.triangle_uvs) > 0
    uv = np.asarray(mesh.triangle_uvs).reshape(-1, 2) if has_uv else None

    mtl_name = f"{name}.mtl"
    mat_name = f"{name}_mat"
    mtl_path = out_path.with_suffix(".mtl")

    # ── OBJ body ──
    lines: list[str] = []
    lines.append("# Rhino")
    lines.append("")
    lines.append(f"mtllib {mtl_name}")
    lines.append(f"usemtl {mat_name}")

    # vertices
    for x, y, z in v:
        lines.append(f"v {x:.16f} {y:.16f} {z:.16f}")

    # texture coords
    if uv is not None:
        for u, w in uv:
            lines.append(f"vt {u:.20f} {w:.20f}")

    # normals
    for x, y, z in n:
        lines.append(f"vn {x:.16f} {y:.16f} {z:.16f}")

    # faces (1-indexed)
    if uv is not None:
        # triangle_uvs: flat [3*nT, 2], row i*3+j is uv of tri i corner j
        # OBJ vt indices are 1-based, sequential
        uv_idx = np.arange(len(uv)).reshape(-1, 3)   # [nT, 3]
        for tri_i, (va, vb, vc) in enumerate(t):
            ua, ub, uc = uv_idx[tri_i] + 1           # 1-based
            lines.append(
                f"f {va+1}/{ua}/{va+1} {vb+1}/{ub}/{vb+1} {vc+1}/{uc}/{vc+1}")
    else:
        # no UV: v//vn
        for va, vb, vc in t:
            lines.append(
                f"f {va+1}//{va+1} {vb+1}//{vb+1} {vc+1}//{vc+1}")

    obj_text = "\r\n".join(lines) + "\r\n"
    out_path.write_text(obj_text, encoding="utf-8")

    # ── MTL ──
    tex_line = ""
    if tex_png is not None and tex_png.exists():
        tex_line = f"map_Kd {tex_png.name}\r\n"

    mtl_text = (
        f"newmtl {mat_name}\r\n"
        f"Ka 1.000 1.000 1.000\r\n"
        f"Kd 1.000 1.000 1.000\r\n"
        f"Ks 0.000 0.000 0.000\r\n"
        f"d 1.0\r\n"
        f"illum 2\r\n"
        f"{tex_line}"
    )
    mtl_path.write_text(mtl_text, encoding="utf-8")
