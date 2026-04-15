"""
Custom insole generator.
Builds a parametric insole mesh based on foot measurements
and biomechanical analysis, then exports a watertight STL.

Accuracy target: 1-2mm dimensional tolerance on length/width.
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


def _foot_outline(t: float, width: float) -> float:
    """
    Parametric foot outline: returns half-width at longitudinal position t ∈ [0, 1].
    Models the natural foot contour (narrow heel, wide forefoot, tapered toes)
    for 1-2mm accuracy on width profile.

    t=0 → heel, t=1 → toe tip
    """
    # Heel region [0, 0.15]: narrow, rounded
    if t < 0.15:
        s = t / 0.15
        return width * 0.35 * np.sqrt(np.clip(1.0 - (1.0 - s) ** 2, 0, 1))

    # Midfoot [0.15, 0.40]: widening through arch
    if t < 0.40:
        s = (t - 0.15) / 0.25
        return width * (0.35 + 0.15 * s)

    # Forefoot [0.40, 0.75]: widest (metatarsal heads)
    if t < 0.75:
        s = (t - 0.40) / 0.35
        # Slight medial/lateral asymmetry for anatomical accuracy
        return width * (0.50 - 0.02 * np.sin(np.pi * s))

    # Toe taper [0.75, 1.0]: narrowing to tip
    s = (t - 0.75) / 0.25
    return width * 0.50 * (1.0 - s ** 1.5)


def _build_insole_profile(
    length: float,
    width: float,
    arch_type: str,
    heel_cup: float = HEEL_CUP_DEPTH_MM,
    forefoot_cushion: float = FOREFOOT_CUSHION_MM,
    thickness: float = INSOLE_THICKNESS_MM,
    n_length: int = 120,
    n_width: int = 60,
) -> trimesh.Trimesh:
    """
    Generate a parametric insole as a displaced grid mesh.

    Improvements over v1 for 1-2mm accuracy:
    - Anatomical foot outline instead of elliptic approximation
    - Higher grid resolution (120×60 vs 80×40)
    - Metatarsal pad region modelling
    - Smooth heel cup transition
    """
    arch_height = ARCH_SUPPORT_MAP.get(arch_type, INSOLE_ARCH_HEIGHT_NORMAL)

    xs = np.linspace(0, length, n_length)
    ys_norm = np.linspace(-0.5, 0.5, n_width)

    vertices_top = []
    vertices_bot = []

    for i, x in enumerate(xs):
        t = x / length  # longitudinal position [0, 1]
        half_w = _foot_outline(t, width)

        for j, yn in enumerate(ys_norm):
            y = yn * 2.0 * half_w  # scale to actual width at this slice

            # --- Lateral mask: smooth rolloff at foot edges ---
            y_ratio = abs(yn)  # 0 at center, 0.5 at edge
            lateral = np.sqrt(np.clip(1.0 - (2.0 * y_ratio) ** 2, 0, 1))

            # --- Heel cup: concave depression in [0, 0.15] ---
            if t < 0.15:
                heel_s = t / 0.15
                heel_z = -heel_cup * (1.0 - heel_s ** 2) * lateral
            else:
                heel_z = 0.0

            # --- Arch support: bell curve in [0.25, 0.55] ---
            # Medially biased (yn < 0 = medial side)
            medial_bias = 1.0 + 0.3 * np.clip(-yn * 2.0, 0, 1)
            arch_z = arch_height * np.exp(-((t - 0.40) ** 2) / (2 * 0.06 ** 2)) * lateral * medial_bias

            # --- Metatarsal pad: subtle dome in [0.55, 0.75] ---
            met_z = 2.0 * np.exp(-((t - 0.65) ** 2) / (2 * 0.04 ** 2)) * lateral

            # --- Forefoot cushion: gentle raise in [0.70, 0.95] ---
            fore_z = forefoot_cushion * np.exp(-((t - 0.82) ** 2) / (2 * 0.08 ** 2)) * lateral

            z_top = heel_z + arch_z + met_z + fore_z + thickness
            z_bot = 0.0

            vertices_top.append([x, y, z_top])
            vertices_bot.append([x, y, z_bot])

    top_verts = np.array(vertices_top, dtype=np.float64)
    bot_verts = np.array(vertices_bot, dtype=np.float64)
    all_verts = np.vstack([top_verts, bot_verts])

    n_v = n_length * n_width

    # Triangulate both surfaces
    faces = []
    for i in range(n_length - 1):
        for j in range(n_width - 1):
            idx = i * n_width + j
            # Top (outward up)
            faces.append([idx, idx + 1, idx + n_width])
            faces.append([idx + 1, idx + n_width + 1, idx + n_width])
            # Bottom (outward down, reverse winding)
            b = idx + n_v
            faces.append([b, b + n_width, b + 1])
            faces.append([b + 1, b + n_width, b + n_width + 1])

    # Stitch edges for watertight mesh
    # Front edge
    for j in range(n_width - 1):
        t_idx = (n_length - 1) * n_width + j
        b_idx = t_idx + n_v
        faces.append([t_idx, b_idx, t_idx + 1])
        faces.append([t_idx + 1, b_idx, b_idx + 1])
    # Back edge
    for j in range(n_width - 1):
        t_idx = j
        b_idx = t_idx + n_v
        faces.append([t_idx, t_idx + 1, b_idx])
        faces.append([t_idx + 1, b_idx + 1, b_idx])
    # Left edge (j=0)
    for i in range(n_length - 1):
        t_idx = i * n_width
        b_idx = t_idx + n_v
        faces.append([t_idx, t_idx + n_width, b_idx])
        faces.append([t_idx + n_width, b_idx + n_width, b_idx])
    # Right edge (j=n_width-1)
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

    # Verify dimensional accuracy
    bb = mesh.bounding_box.extents
    log.info(
        "Insole mesh: %d verts, %d faces, watertight=%s, "
        "actual dims: %.1f x %.1f x %.1f mm (target: %.1f x %.1f)",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
        bb[0], bb[1], bb[2], length, width,
    )

    # FIX: Verify 1-2mm accuracy on length/width
    length_error = abs(bb[0] - length)
    width_error = abs(bb[1] - width)
    if length_error > 2.0:
        log.warning("Length error %.1f mm exceeds 2mm target!", length_error)
    if width_error > 2.0:
        log.warning("Width error %.1f mm exceeds 2mm target!", width_error)

    return mesh


def generate_insole(
    landmarks: FootLandmarks,
    arch_type: str,
    output_path: Path,
) -> Path:
    """Generate a custom insole STL from foot landmarks and arch classification."""
    log.info(
        "Generating insole: length=%.1f mm, width=%.1f mm, arch=%s",
        landmarks.foot_length_mm, landmarks.foot_width_mm, arch_type,
    )

    # FIX: Clamp to sane ranges for manufacturing
    length = np.clip(landmarks.foot_length_mm, 150.0, 350.0)
    width = np.clip(landmarks.foot_width_mm, 60.0, 130.0)

    mesh = _build_insole_profile(
        length=length,
        width=width,
        arch_type=arch_type,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="stl")

    log.info("STL exported to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
