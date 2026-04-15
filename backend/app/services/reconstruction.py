"""
3-D Reconstruction service.
Generates a watertight mesh from a cleaned point cloud.
Uses trimesh + scipy for Poisson-like surface reconstruction.
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay

from app.config import POISSON_DEPTH
from app.utils.logger import get_logger
from app.services.geometry_utils import compute_normals_from_points

log = get_logger(__name__)


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate normals via local PCA on k-nearest neighbours."""
    log.info("Estimating normals (k=%d) on %d points", k, len(points))
    # FIX: Clamp k to available points
    effective_k = min(k, len(points))
    if effective_k < 2:
        log.warning("Too few points for normal estimation — using up-facing normals")
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals.astype(np.float32)
    normals = compute_normals_from_points(points, k=effective_k)
    log.info("Normals estimated for %d points", len(points))
    return normals


def reconstruct_mesh(
    points: np.ndarray,
    normals: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Reconstruct a watertight surface from points.
    Strategy:
    1. Alpha shape (Delaunay + edge-length filtering).
    2. Convex hull fallback.
    """
    log.info("Reconstructing mesh from %d points ...", len(points))

    # FIX: Need at least 4 points for 3D reconstruction
    if len(points) < 4:
        log.warning("Only %d points — generating minimal tetrahedron", len(points))
        return _generate_minimal_mesh(points)

    try:
        mesh = _alpha_shape_reconstruction(points)
        if mesh is not None and len(mesh.faces) > 10:
            log.info("Alpha-shape reconstruction succeeded (%d faces)", len(mesh.faces))
            return mesh
    except Exception as exc:
        log.warning("Alpha-shape failed: %s — falling back to convex hull", exc)

    # Fallback: convex hull (always watertight)
    log.info("Using convex hull reconstruction")
    try:
        hull = ConvexHull(points)
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=hull.simplices,
            process=True,
        )
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception as exc:
        log.warning("ConvexHull failed: %s — generating minimal mesh", exc)
        return _generate_minimal_mesh(points)

    log.info(
        "Reconstructed mesh: %d vertices, %d faces",
        len(mesh.vertices), len(mesh.faces),
    )
    return mesh


def _generate_minimal_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """Create a valid mesh from very few points by generating a bounding box."""
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    dims = bb_max - bb_min
    # Ensure non-zero dimensions
    for i in range(3):
        if dims[i] < 1e-6:
            bb_min[i] -= 0.5
            bb_max[i] += 0.5
    mesh = trimesh.creation.box(
        extents=bb_max - bb_min,
        transform=trimesh.transformations.translation_matrix((bb_min + bb_max) / 2),
    )
    log.info("Generated bounding box mesh: %d verts, %d faces", len(mesh.vertices), len(mesh.faces))
    return mesh


def _alpha_shape_reconstruction(
    points: np.ndarray,
    alpha: float | None = None,
) -> trimesh.Trimesh | None:
    """Compute an alpha shape from 3D points using Delaunay triangulation."""
    if len(points) < 4:
        return None

    try:
        tri = Delaunay(points)
    except Exception:
        return None

    if alpha is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(2, len(points)))
        if dists.ndim == 1:
            mean_dist = dists.mean()
        else:
            mean_dist = dists[:, 1].mean() if dists.shape[1] > 1 else dists.mean()
        if mean_dist < 1e-10:
            return None
        alpha = 1.0 / (mean_dist * 3.0)

    valid_faces = set()
    for simplex in tri.simplices:
        pts = points[simplex]
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(pts[i] - pts[j]))
        max_edge = max(edges)
        if max_edge < 1.0 / alpha:
            for i in range(4):
                face = tuple(sorted([simplex[j] for j in range(4) if j != i]))
                valid_faces.add(face)

    if not valid_faces:
        return None

    faces = np.array(list(valid_faces))
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attempt to make the mesh watertight."""
    if mesh.is_watertight:
        log.info("Mesh is already watertight [OK]")
        return mesh

    log.warning("Mesh is NOT watertight — attempting repair ...")
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if mesh.is_watertight:
        log.info("Mesh repaired successfully [OK]")
    else:
        log.warning("Mesh still not watertight after repair — proceeding anyway")
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 3) -> trimesh.Trimesh:
    """Apply Laplacian smoothing via trimesh."""
    # FIX: Only smooth if enough geometry exists
    if len(mesh.vertices) < 10 or len(mesh.faces) < 10:
        log.warning("Too few faces for smoothing — skipping")
        return mesh
    log.info("Smoothing mesh (%d iterations)", iterations)
    trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
    return mesh

