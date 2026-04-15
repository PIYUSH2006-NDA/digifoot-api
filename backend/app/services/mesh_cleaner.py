"""
Mesh cleaning service.
Removes noise, degenerate geometry, and the ground plane.
Uses trimesh + scipy instead of Open3D for maximum compatibility.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from app.config import (
    VOXEL_DOWNSAMPLE_SIZE,
    STATISTICAL_NB_NEIGHBORS,
    STATISTICAL_STD_RATIO,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# Minimum point count below which cleaning is skipped to avoid crashes
MIN_POINTS_FOR_CLEANING = 20


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a triangle mesh from .obj or .ply."""
    log.info("Loading mesh from %s", path)
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single mesh, got {type(mesh)} from {path}")
    if len(mesh.vertices) == 0:
        raise ValueError(f"Empty or unreadable mesh at {path}")
    log.info(
        "Loaded mesh: %d vertices, %d faces",
        len(mesh.vertices),
        len(mesh.faces),
    )
    return mesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Full cleaning pipeline:
    1. Merge duplicate vertices.
    2. Remove degenerate faces.
    3. Remove unreferenced vertices.
    4. Statistical outlier removal (only if enough points).
    """
    log.info("Cleaning mesh ...")

    # Merge close vertices
    mesh.merge_vertices()

    # Remove degenerate (zero-area) faces
    face_areas = mesh.area_faces
    valid = face_areas > 1e-10
    if not valid.all():
        removed = (~valid).sum()
        mesh.update_faces(valid)
        mesh.remove_unreferenced_vertices()
        log.info("Removed %d degenerate faces", removed)

    # FIX: Skip statistical outlier removal if too few vertices
    points = np.asarray(mesh.vertices)
    if len(points) < MIN_POINTS_FOR_CLEANING:
        log.warning(
            "Only %d vertices — skipping statistical outlier removal (need >=%d)",
            len(points), MIN_POINTS_FOR_CLEANING,
        )
        return mesh

    clean_mask = _statistical_outlier_mask(
        points,
        nb_neighbors=min(STATISTICAL_NB_NEIGHBORS, len(points) - 1),
        std_ratio=STATISTICAL_STD_RATIO,
    )
    if not clean_mask.all():
        removed = (~clean_mask).sum()
        vertex_map = np.full(len(points), -1, dtype=int)
        new_indices = np.arange(clean_mask.sum())
        vertex_map[clean_mask] = new_indices
        face_keep = np.all(clean_mask[mesh.faces], axis=1)
        new_faces = vertex_map[mesh.faces[face_keep]]
        mesh = trimesh.Trimesh(
            vertices=points[clean_mask],
            faces=new_faces,
            process=True,
        )
        log.info("Removed %d outlier vertices", removed)

    log.info(
        "Cleaned mesh: %d vertices, %d faces",
        len(mesh.vertices),
        len(mesh.faces),
    )
    return mesh


def _statistical_outlier_mask(
    points: np.ndarray,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Return boolean mask of inlier points using statistical outlier removal."""
    # FIX: Clamp nb_neighbors to available points
    k = min(nb_neighbors + 1, len(points))
    if k < 2:
        return np.ones(len(points), dtype=bool)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    mean_dists = distances[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    if global_std < 1e-10:
        return np.ones(len(points), dtype=bool)
    threshold = global_mean + std_ratio * global_std
    return mean_dists < threshold


def mesh_to_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract vertex positions as Nx3 array."""
    return np.asarray(mesh.vertices, dtype=np.float64)


def remove_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 5.0,  # mm
    num_iterations: int = 1000,
) -> np.ndarray:
    """
    Detect and remove the dominant ground plane using RANSAC.
    Returns the point array with ground points removed.
    """
    # FIX: Skip if too few points
    if len(points) < 10:
        log.warning("Too few points (%d) for ground removal — skipping", len(points))
        return points

    log.info("Detecting ground plane (RANSAC) ...")

    best_inliers = np.array([], dtype=int)
    n = len(points)

    rng = np.random.default_rng(42)
    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        dists = np.abs(points @ normal + d)
        inliers = np.where(dists < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    log.info("Ground plane inliers: %d / %d", len(best_inliers), n)

    # FIX: Don't remove ground if it would leave < 4 points
    remaining = n - len(best_inliers)
    if remaining < 4:
        log.warning("Ground removal would leave only %d points — skipping", remaining)
        return points

    mask = np.ones(n, dtype=bool)
    mask[best_inliers] = False
    result = points[mask]
    log.info("Points after ground removal: %d", len(result))
    return result


def downsample_points(
    points: np.ndarray,
    voxel_size: float = VOXEL_DOWNSAMPLE_SIZE,
) -> np.ndarray:
    """Voxel-grid down-sampling on raw points."""
    if len(points) < 10:
        log.warning("Too few points for downsampling — skipping")
        return points

    log.info("Down-sampling with voxel size %.2f mm", voxel_size)
    quantised = np.floor(points / max(voxel_size, 1e-6)).astype(np.int64)
    _, unique_idx = np.unique(quantised, axis=0, return_index=True)
    result = points[unique_idx]
    log.info("Points after down-sample: %d -> %d", len(points), len(result))
    return result
