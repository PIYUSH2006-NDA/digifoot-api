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
    log.info("Estimating normals (k=%d)", k)
    normals = compute_normals_from_points(points, k=k)
    log.info("Normals estimated for %d points", len(points))
    return normals


def reconstruct_mesh(
    points: np.ndarray,
    normals: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Reconstruct a watertight surface from points.
    
    Strategy:
    1. Use Ball Pivoting-like approach via alpha shape (Delaunay + filtering).
    2. Fall back to convex hull if alpha shape fails.
    3. Ensure the result is watertight.
    """
    log.info("Reconstructing mesh from %d points ...", len(points))

    try:
        # Attempt alpha-shape reconstruction via trimesh
        mesh = _alpha_shape_reconstruction(points)
        if mesh is not None and len(mesh.faces) > 100:
            log.info("Alpha-shape reconstruction succeeded")
            return mesh
    except Exception as exc:
        log.warning("Alpha-shape failed: %s - falling back to convex hull", exc)

    # Fallback: convex hull (always watertight)
    log.info("Using convex hull reconstruction")
    hull = ConvexHull(points)
    mesh = trimesh.Trimesh(
        vertices=points,
        faces=hull.simplices,
        process=True,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    log.info(
        "Reconstructed mesh: %d vertices, %d faces",
        len(mesh.vertices), len(mesh.faces),
    )
    return mesh


def _alpha_shape_reconstruction(
    points: np.ndarray,
    alpha: float | None = None,
) -> trimesh.Trimesh | None:
    """
    Compute an alpha shape from 3D points using Delaunay triangulation.
    Edges longer than 1/alpha are removed.
    """
    if len(points) < 4:
        return None

    try:
        tri = Delaunay(points)
    except Exception:
        return None

    if alpha is None:
        # Adaptive alpha: use mean nearest-neighbour distance × 3
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        alpha = 1.0 / (dists[:, 1].mean() * 3.0)

    # Filter tetrahedra by circumradius
    valid_faces = set()
    for simplex in tri.simplices:
        pts = points[simplex]
        # Compute max edge length
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(pts[i] - pts[j]))
        max_edge = max(edges)
        if max_edge < 1.0 / alpha:
            # Add all 4 triangular faces of the tetrahedron
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

    log.warning("Mesh is NOT watertight - attempting repair ...")
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if mesh.is_watertight:
        log.info("Mesh repaired successfully [OK]")
    else:
        log.warning("Mesh still not watertight after repair - proceeding anyway")
    return mesh


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
) -> trimesh.Trimesh:
    """Apply Laplacian smoothing via trimesh."""
    log.info("Smoothing mesh (%d iterations)", iterations)
    trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
    return mesh
