"""
Custom insole generator.
Builds a parametric insole mesh based on foot measurements
and biomechanical analysis, then exports a watertight STL.
"""

from pathlib import Path

import numpy as np
import trimesh

from app.config import (
    INSOLE_THICKNESS_MM,
    INSOLE_ARCH_HEIGHT_FLAT,
    INSOLE_ARCH_HEIGHT_NORMAL,
    INSOLE_ARCH_HEIGHT_HIGH,
    HEEL_CUP_DEPTH_MM,
    FOREFOOT_CUSHION_MM,
)
from app.services.landmark_detector import FootLandmarks
from app.utils.logger import get_logger

log = get_logger(__name__)

ARCH_SUPPORT_MAP = {
    "flat": INSOLE_ARCH_HEIGHT_FLAT,
    "normal": INSOLE_ARCH_HEIGHT_NORMAL,
    "high": INSOLE_ARCH_HEIGHT_HIGH,
}


def _build_insole_profile(
    length: float,
    width: float,
    arch_type: str,
    heel_cup: float = HEEL_CUP_DEPTH_MM,
    forefoot_cushion: float = FOREFOOT_CUSHION_MM,
    thickness: float = INSOLE_THICKNESS_MM,
    n_length: int = 80,
    n_width: int = 40,
) -> trimesh.Trimesh:
    """
    Generate a parametric insole as a displaced grid mesh.

    The insole is modelled as a 2.5-D surface where the Z-height at each
    (x, y) grid point is controlled by parametric curves for:
      • Heel cup (concave at the rear)
      • Arch support (convex in the midfoot, height depends on arch type)
      • Forefoot cushion (gentle padding zone)
    A bottom plate is added to make the mesh manifold/watertight.
    """
    arch_height = ARCH_SUPPORT_MAP.get(arch_type, INSOLE_ARCH_HEIGHT_NORMAL)

    xs = np.linspace(0, length, n_length)
    ys = np.linspace(-width / 2, width / 2, n_width)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")

    # Normalised longitudinal position [0, 1]
    t = xg / length

    # --- Longitudinal profile ---
    # Heel cup: concave in [0, 0.15]
    heel_curve = np.where(
        t < 0.15,
        -heel_cup * np.cos(np.pi * t / 0.15) + heel_cup,
        0.0,
    )
    # Arch support: bell curve in [0.25, 0.55]
    arch_curve = arch_height * np.exp(-((t - 0.40) ** 2) / (2 * 0.06 ** 2))
    # Forefoot cushion: slight raise in [0.70, 0.95]
    fore_curve = forefoot_cushion * np.exp(-((t - 0.82) ** 2) / (2 * 0.08 ** 2))

    # --- Lateral profile (elliptic for foot contour) ---
    lateral = np.sqrt(np.clip(1.0 - (2 * yg / width) ** 2, 0, 1))

    z_top = (heel_curve + arch_curve + fore_curve) * lateral + thickness
    z_bot = np.zeros_like(z_top)  # flat bottom

    # Build vertices for top and bottom surfaces
    top_verts = np.stack([xg, yg, z_top], axis=-1).reshape(-1, 3)
    bot_verts = np.stack([xg, yg, z_bot], axis=-1).reshape(-1, 3)
    all_verts = np.vstack([top_verts, bot_verts])

    n_v = n_length * n_width  # vertices per surface

    # Triangulate each surface as a grid
    faces = []
    for i in range(n_length - 1):
        for j in range(n_width - 1):
            idx = i * n_width + j
            # Top surface (outward normal up)
            faces.append([idx, idx + 1, idx + n_width])
            faces.append([idx + 1, idx + n_width + 1, idx + n_width])
            # Bottom surface (outward normal down, reverse winding)
            b = idx + n_v
            faces.append([b, b + n_width, b + 1])
            faces.append([b + 1, b + n_width, b + n_width + 1])

    # --- Stitch edges to make watertight ---
    # Front edge (i = n_length - 1)
    for j in range(n_width - 1):
        t_idx = (n_length - 1) * n_width + j
        b_idx = t_idx + n_v
        faces.append([t_idx, b_idx, t_idx + 1])
        faces.append([t_idx + 1, b_idx, b_idx + 1])

    # Back edge (i = 0)
    for j in range(n_width - 1):
        t_idx = j
        b_idx = t_idx + n_v
        faces.append([t_idx, t_idx + 1, b_idx])
        faces.append([t_idx + 1, b_idx + 1, b_idx])

    # Left edge (j = 0)
    for i in range(n_length - 1):
        t_idx = i * n_width
        b_idx = t_idx + n_v
        faces.append([t_idx, t_idx + n_width, b_idx])
        faces.append([t_idx + n_width, b_idx + n_width, b_idx])

    # Right edge (j = n_width - 1)
    for i in range(n_length - 1):
        t_idx = i * n_width + (n_width - 1)
        b_idx = t_idx + n_v
        faces.append([t_idx, b_idx, t_idx + n_width])
        faces.append([t_idx + n_width, b_idx, b_idx + n_width])

    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=all_verts, faces=faces, process=True)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    log.info(
        "Insole mesh: %d verts, %d faces, watertight=%s",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )
    return mesh


def generate_insole(
    landmarks: FootLandmarks,
    arch_type: str,
    output_path: Path,
) -> Path:
    """
    Generate a custom insole STL from foot landmarks and arch classification.
    """
    log.info(
        "Generating insole: length=%.1f mm, width=%.1f mm, arch=%s",
        landmarks.foot_length_mm,
        landmarks.foot_width_mm,
        arch_type,
    )

    mesh = _build_insole_profile(
        length=landmarks.foot_length_mm,
        width=landmarks.foot_width_mm,
        arch_type=arch_type,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="stl")

    log.info("STL exported to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
