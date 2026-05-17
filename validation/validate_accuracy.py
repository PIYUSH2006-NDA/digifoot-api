"""
validate_accuracy.py
Measure surface error of pipeline output vs clinic ground-truth scan.

Usage:
    python validate_accuracy.py --pred foot.obj --gt clinic_001.ply --report report.json

What it does:
    1. load both meshes
    2. coarse align (PCA canonical)
    3. fine align (multi-scale ICP)
    4. compute per-vertex distance: pred -> closest point on gt
    5. report:
        - mean error (mm)
        - median error (mm)
        - p95 error (mm)
        - max error (mm)
        - F-score @ 1mm, 2mm
    6. save heatmap PLY (color-coded by error)
"""
import argparse, json, numpy as np, open3d as o3d
from pathlib import Path


def pca_align(src_pts, dst_pts):
    def axes(p):
        c = p.mean(0); p2 = p - c
        cov = np.cov(p2.T); w, Q = np.linalg.eigh(cov)
        Q = Q[:, np.argsort(-w)]
        if np.linalg.det(Q) < 0: Q[:, 2] *= -1
        return c, Q
    cs, Qs = axes(src_pts)
    cd, Qd = axes(dst_pts)
    R = Qd @ Qs.T
    t = cd - R @ cs
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def icp_refine(src_pcd, dst_pcd, init):
    T = init
    for vox in [0.01, 0.005, 0.002, 0.001]:
        a = src_pcd.voxel_down_sample(vox); a.estimate_normals()
        b = dst_pcd.voxel_down_sample(vox); b.estimate_normals()
        reg = o3d.pipelines.registration.registration_icp(
            a, b, vox * 2, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = reg.transformation
    return T


def main(a):
    pred = o3d.io.read_triangle_mesh(a.pred)
    gt = o3d.io.read_triangle_mesh(a.gt)

    # convert mm/m if needed: assume mm by default
    if a.pred_unit == "m":
        pred.scale(1000.0, center=(0, 0, 0))
    if a.gt_unit == "m":
        gt.scale(1000.0, center=(0, 0, 0))

    pred_pts = np.asarray(pred.vertices)
    gt_pts = np.asarray(gt.vertices)

    # PCA coarse align
    T0 = pca_align(pred_pts, gt_pts)
    pred.transform(T0)

    # ICP fine align
    pred_pcd = pred.sample_points_uniformly(100_000)
    gt_pcd = gt.sample_points_uniformly(100_000)
    T1 = icp_refine(pred_pcd, gt_pcd, np.eye(4))
    pred.transform(T1)

    # distance: each pred vertex -> closest gt surface
    gt_legacy = o3d.t.geometry.TriangleMesh.from_legacy(gt)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(gt_legacy)
    pv = np.asarray(pred.vertices, dtype=np.float32)
    qry = o3d.core.Tensor(pv, dtype=o3d.core.Dtype.Float32)
    dists = scene.compute_distance(qry).numpy()   # mm

    rep = {
        "n_verts": int(len(pv)),
        "mean_mm": float(np.mean(dists)),
        "median_mm": float(np.median(dists)),
        "p95_mm": float(np.percentile(dists, 95)),
        "max_mm": float(np.max(dists)),
        "f_at_1mm": float(np.mean(dists < 1.0)),
        "f_at_2mm": float(np.mean(dists < 2.0)),
        "f_at_3mm": float(np.mean(dists < 3.0)),
    }
    Path(a.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

    # heatmap
    if a.heatmap:
        cmin, cmax = 0.0, 3.0
        d_norm = np.clip((dists - cmin) / (cmax - cmin), 0, 1)
        # blue (good) -> red (bad)
        colors = np.stack([d_norm, 1 - d_norm, np.zeros_like(d_norm) + 0.2], -1)
        pred.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(a.heatmap, pred)
        print(f"heatmap -> {a.heatmap}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pipeline output (.obj/.ply)")
    ap.add_argument("--gt", required=True, help="ground-truth clinic scan (.ply)")
    ap.add_argument("--report", default="accuracy.json")
    ap.add_argument("--heatmap", default="heatmap.ply", help="set empty to skip")
    ap.add_argument("--pred-unit", choices=["mm", "m"], default="mm")
    ap.add_argument("--gt-unit", choices=["mm", "m"], default="mm")
    main(ap.parse_args())
