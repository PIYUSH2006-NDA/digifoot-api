"""
build_template.py
Turn a raw foot scan (any source) into a clean foot_template.ply
ready for synthetic_gen.py.

Usage:
    python build_template.py --in raw_foot.obj --out assets/foot_template.ply

What it does:
    1. load raw mesh
    2. center + canonical align (long axis = X, sole down = Z)
    3. drop everything below ankle (cut at z = ankle_height)
    4. largest connected component (drop floor / hands)
    5. fill holes (Poisson)
    6. taubin smooth (volume-preserving)
    7. isotropic remesh to ~10k vertices
    8. compute clean normals
    9. save .ply with normals
"""
import argparse, numpy as np, open3d as o3d
from pathlib import Path


def canonical_align(mesh):
    v = np.asarray(mesh.vertices)
    c = v.mean(0); v = v - c
    cov = np.cov(v.T)
    w, Q = np.linalg.eigh(cov)
    Q = Q[:, np.argsort(-w)]                      # cols: long, mid, short
    if np.linalg.det(Q) < 0: Q[:, 2] *= -1
    v = v @ Q
    # ensure sole down: most points should be at high Z if foot upside-down -> flip
    if (v[:, 2] > v[:, 2].mean()).sum() > (v[:, 2] < v[:, 2].mean()).sum():
        v[:, 2] *= -1
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def cut_above_ankle(mesh, ankle_frac=0.4):
    """drop top fraction above ankle (relative to bbox height)."""
    v = np.asarray(mesh.vertices)
    z_lo, z_hi = v[:, 2].min(), v[:, 2].max()
    cut = z_lo + (z_hi - z_lo) * ankle_frac
    keep_v = v[:, 2] < cut
    # remove triangles that have any vertex above cut
    t = np.asarray(mesh.triangles)
    keep_t = keep_v[t].all(1)
    mesh.remove_triangles_by_mask(~keep_t)
    mesh.remove_unreferenced_vertices()
    return mesh


def largest_component(mesh):
    cl, _, areas = mesh.cluster_connected_triangles()
    cl = np.asarray(cl); areas = np.asarray(areas)
    keep = cl == areas.argmax()
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_holes_and_smooth(mesh, depth=9):
    pcd = mesh.sample_points_poisson_disk(50_000)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    p, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    p.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
    p = p.filter_smooth_taubin(number_of_iterations=10)
    p.compute_vertex_normals()
    return p


def remesh(mesh, target_verts=10_000):
    target_tris = target_verts * 2
    m = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    return m


def main(a):
    inp = Path(a.inp); out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("load:", inp)
    m = o3d.io.read_triangle_mesh(str(inp))
    if len(m.vertices) == 0: raise SystemExit("empty mesh")
    print(f"  raw: {len(m.vertices)} verts, {len(m.triangles)} tris")

    print("align canonical")
    m = canonical_align(m)

    if not a.keep_above:
        print(f"cut above ankle (frac={a.ankle})")
        m = cut_above_ankle(m, a.ankle)

    print("largest component")
    m = largest_component(m)

    print("fill holes + smooth (poisson depth=9)")
    m = fill_holes_and_smooth(m)

    print(f"remesh to ~{a.verts} verts")
    m = remesh(m, a.verts)

    # rescale to ~250mm length (canonical)
    v = np.asarray(m.vertices)
    L = v[:, 0].ptp()
    if L > 0:
        scale = 250.0 / L
        v = v * scale
        m.vertices = o3d.utility.Vector3dVector(v)
        print(f"rescaled by {scale:.4f} -> length {v[:,0].ptp():.1f} mm")

    o3d.io.write_triangle_mesh(str(out), m, write_ascii=False)
    print(f"saved -> {out}  ({len(m.vertices)} verts, {len(m.triangles)} tris)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="raw foot mesh (obj/ply/stl)")
    ap.add_argument("--out", required=True, help="output foot_template.ply")
    ap.add_argument("--ankle", type=float, default=0.4,
                    help="ankle cut fraction of bbox height (0=heel, 1=top)")
    ap.add_argument("--verts", type=int, default=10_000)
    ap.add_argument("--keep-above", action="store_true",
                    help="don't cut above ankle (use if scan is foot-only)")
    main(ap.parse_args())
