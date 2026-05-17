"""
backend/app/recon/ml_refine.py
Wraps PointNet2FootRefine inference.
Input  : Open3D mesh
Output : Open3D mesh refined via Poisson re-mesh
"""
import numpy as np
import open3d as o3d
import torch

from ..ml.model_loader import ModelRegistry, DEVICE

_NPTS = 16_384


def _normalize(pts: np.ndarray):
    c = pts.mean(0)
    s = float(np.linalg.norm(pts - c, axis=1).max() + 1e-8)
    return (pts - c) / s, c, s


def refine_with_pointnet2(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """ML completion + normal refine. Graceful no-op if weights missing."""
    try:
        model = ModelRegistry.get("pointnet2_refine")
    except Exception as e:
        print(f"[ml_refine] skip ({e})")
        return mesh

    pcd = mesh.sample_points_poisson_disk(_NPTS)
    pts = np.asarray(pcd.points, dtype=np.float32)
    norm, c, s = _normalize(pts)

    with torch.no_grad():
        x = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
        disp, n_pred = model(x)
        disp = disp.cpu().numpy()[0]
        n_pred = n_pred.cpu().numpy()[0]

    pts_ref = (norm + disp) * s + c
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(pts_ref)
    pcd_ref.normals = o3d.utility.Vector3dVector(n_pred)

    p, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_ref, depth=10)
    dens = np.asarray(dens)
    p.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
    p.compute_vertex_normals()
    return p
