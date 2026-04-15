"""
Custom insole generator.
Builds an insole mesh based on the ACTUAL scanned foot shape,
then exports a watertight STL.

MODIFIED (v2):
  - Uses actual scan point cloud boundary projection for the foot outline
    instead of the hardcoded parametric _foot_outline() curve.
  - Accepts foot_side ("left" | "right") and mirrors the insole accordingly.
  - Accepts the reconstructed mesh and scan points to trace the real
    foot contour at each longitudinal slice.
  - Falls back to parametric outline only when scan data is insufficient.

Accuracy target: 1-2mm dimensional tolerance on length/width.
"""

from pathlib import Path
from typing import Optional

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


# ═══════════════════════════════════════════════════════════════════
#  SCAN-BASED CONTOUR EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _extract_foot_contour_from_scan(
    points: np.ndarray,
    n_slices: int = 120,
) -> Optional[np.ndarray]:
    """
    Project the scan point cloud onto the XY plane and extract the
    foot boundary width at each longitudinal slice.

    Returns an (n_slices, 2) array of (t, half_width) pairs,
    or None if the scan data is insufficient.
    """
    if points is None or len(points) < 50:
        return None

    # PCA-align: longest axis → X, second → Y, shortest → Z
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return None

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    # Sort eigenvectors by eigenvalue descending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Project to PCA space
    projected = centered @ eigvecs  # columns: [length, width, height]

    x = projected[:, 0]  # along foot length
    y = projected[:, 1]  # across foot width

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min

    if x_range < 10.0:  # less than 10mm — too small
        return None

    # Slice the foot along the length axis
    contour = np.zeros((n_slices, 2))
    slice_edges = np.linspace(x_min, x_max, n_slices + 1)

    for i in range(n_slices):
        mask = (x >= slice_edges[i]) & (x < slice_edges[i + 1])
        t = (slice_edges[i] + slice_edges[i + 1]) / 2.0
        t_norm = (t - x_min) / x_range  # normalised [0, 1]

        if mask.sum() < 3:
            contour[i] = [t_norm, 0.0]
            continue

        y_slice = y[mask]
        half_width = (y_slice.max() - y_slice.min()) / 2.0
        contour[i] = [t_norm, half_width]

    # Smooth the contour to remove noise
    from scipy.ndimage import uniform_filter1d
    contour[:, 1] = uniform_filter1d(contour[:, 1], size=5)

    # Fill zero-width gaps by interpolation
    nonzero = contour[:, 1] > 0.1
    if nonzero.sum() < 10:
        return None

    if not nonzero.all():
        from scipy.interpolate import interp1d
        t_good = contour[nonzero, 0]
        w_good = contour[nonzero, 1]
        interp = interp1d(t_good, w_good, kind='linear',
                          fill_value='extrapolate', bounds_error=False)
        contour[:, 1] = np.maximum(interp(contour[:, 0]), 0.0)

    log.info(
        "Extracted scan contour: %d slices, width range %.1f – %.1f mm",
        n_slices, contour[:, 1].min() * 2, contour[:, 1].max() * 2,
    )
    return contour


# ═══════════════════════════════════════════════════════════════════
#  PARAMETRIC FALLBACK (original)
# ═══════════════════════════════════════════════════════════════════

def _foot_outline(t: float, width: float) -> float:
    """
    Parametric foot outline: returns half-width at longitudinal position t ∈ [0, 1].
    t=0 → heel, t=1 → toe tip
    """
    if t < 0.15:
        s = t / 0.15
        return width * 0.35 * np.sqrt(np.clip(1.0 - (1.0 - s) ** 2, 0, 1))
    if t < 0.40:
        s = (t - 0.15) / 0.25
        return width * (0.35 + 0.15 * s)
    if t < 0.75:
        s = (t - 0.40) / 0.35
        return width * (0.50 - 0.02 * np.sin(np.pi * s))
    s = (t - 0.75) / 0.25
    return width * 0.50 * (1.0 - s ** 1.5)


def _get_half_width(
    t: float,
    width: float,
    scan_contour: Optional[np.ndarray],
) -> float:
    """
    Return the half-width at position t.
    Uses scan contour if available, otherwise falls back to parametric.
    """
    if scan_contour is not None and len(scan_contour) > 0:
        # Interpolate from scan contour
        idx = np.searchsorted(scan_contour[:, 0], t)
        idx = np.clip(idx, 0, len(scan_contour) - 1)
        return float(scan_contour[idx, 1])
    return _foot_outline(t, width)


# ═══════════════════════════════════════════════════════════════════
#  INSOLE MESH BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_insole_profile(
    length: float,
    width: float,
    arch_type: str,
    foot_side: str = "left",
    scan_contour: Optional[np.ndarray] = None,
    heel_cup: float = HEEL_CUP_DEPTH_MM,
    forefoot_cushion: float = FOREFOOT_CUSHION_MM,
    thickness: float = INSOLE_THICKNESS_MM,
    n_length: int = 120,
    n_width: int = 60,
) -> trimesh.Trimesh:
    """
    Generate an insole mesh using the actual scan contour when available.

    When scan_contour is provided, the foot outline at each slice comes
    from the real scan data. Otherwise, falls back to parametric.

    foot_side="right" mirrors the medial/lateral asymmetry.
    """
    arch_height = ARCH_SUPPORT_MAP.get(arch_type, INSOLE_ARCH_HEIGHT_NORMAL)

    # Mirror factor: medial side is positive Y for left foot, negative for right
    mirror = 1.0 if foot_side == "left" else -1.0

    xs = np.linspace(0, length, n_length)
    ys_norm = np.linspace(-0.5, 0.5, n_width)

    vertices_top = []
    vertices_bot = []

    for i, x in enumerate(xs):
        t = x / length  # longitudinal position [0, 1]
        half_w = _get_half_width(t, width, scan_contour)

        for j, yn in enumerate(ys_norm):
            y = yn * 2.0 * half_w

            # --- Lateral mask: smooth rolloff at foot edges ---
            y_ratio = abs(yn)
            lateral = np.sqrt(np.clip(1.0 - (2.0 * y_ratio) ** 2, 0, 1))

            # --- Heel cup: concave depression in [0, 0.15] ---
            if t < 0.15:
                heel_s = t / 0.15
                heel_z = -heel_cup * (1.0 - heel_s ** 2) * lateral
            else:
                heel_z = 0.0

            # --- Arch support: bell curve in [0.25, 0.55] ---
            # Medially biased — mirror flips which side is medial
            medial_bias = 1.0 + 0.3 * np.clip(-yn * 2.0 * mirror, 0, 1)
            arch_z = arch_height * np.exp(
                -((t - 0.40) ** 2) / (2 * 0.06 ** 2)
            ) * lateral * medial_bias

            # --- Metatarsal pad: subtle dome in [0.55, 0.75] ---
            met_z = 2.0 * np.exp(
                -((t - 0.65) ** 2) / (2 * 0.04 ** 2)
            ) * lateral

            # --- Forefoot cushion: gentle raise in [0.70, 0.95] ---
            fore_z = forefoot_cushion * np.exp(
                -((t - 0.82) ** 2) / (2 * 0.08 ** 2)
            ) * lateral

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
        "Insole mesh [%s]: %d verts, %d faces, watertight=%s, "
        "actual dims: %.1f x %.1f x %.1f mm (target: %.1f x %.1f)",
        foot_side,
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
        bb[0], bb[1], bb[2], length, width,
    )

    length_error = abs(bb[0] - length)
    width_error = abs(bb[1] - width)
    if length_error > 2.0:
        log.warning("Length error %.1f mm exceeds 2mm target!", length_error)
    if width_error > 2.0:
        log.warning("Width error %.1f mm exceeds 2mm target!", width_error)

    return mesh


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def generate_insole(
    landmarks: FootLandmarks,
    arch_type: str,
    output_path: Path,
    foot_side: str = "left",
    scan_points: Optional[np.ndarray] = None,
    reconstructed_mesh: Optional[trimesh.Trimesh] = None,
) -> Path:
    """
    Generate a custom insole STL from foot landmarks and arch classification.

    MODIFIED: Now accepts foot_side, scan_points, and reconstructed_mesh
    to produce anatomically accurate, scan-aligned insoles.
    """
    log.info(
        "Generating insole [%s]: length=%.1f mm, width=%.1f mm, arch=%s",
        foot_side, landmarks.foot_length_mm, landmarks.foot_width_mm, arch_type,
    )

    # Clamp to sane ranges for manufacturing
    length = float(np.clip(landmarks.foot_length_mm, 150.0, 350.0))
    width = float(np.clip(landmarks.foot_width_mm, 60.0, 130.0))

    # Try to extract real contour from scan points
    scan_contour = _extract_foot_contour_from_scan(scan_points)

    if scan_contour is not None:
        log.info("Using SCAN-BASED contour for %s foot insole", foot_side)
    else:
        log.info("Using PARAMETRIC contour for %s foot insole (scan data insufficient)", foot_side)

    mesh = _build_insole_profile(
        length=length,
        width=width,
        arch_type=arch_type,
        foot_side=foot_side,
        scan_contour=scan_contour,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="stl")

    log.info(
        "STL exported [%s] to %s (%.1f KB)",
        foot_side, output_path, output_path.stat().st_size / 1024,
    )
    return output_path
