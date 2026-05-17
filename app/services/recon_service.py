"""
backend/app/services/recon_service.py

Reconstruction pipeline for v6 TrueDepth scans.

Inputs:
  - cloud.ply (live-fused point cloud with normals + confidence)
  - meta.json (poses, intrinsics, covered_octants, tracking confidence)
  - depth/*.bin + depth/*.txt  (raw depth maps for additional fusion)
  - rgb/*.jpg                  (sparse reference RGB for optional texturing)

Pipeline:
  1) load cloud + filter by confidence
  2) statistical outlier removal
  3) align to canonical frame (longest axis = +X toe direction)
  4) optional dense fusion from depth maps if cloud sparse
  5) Poisson surface reconstruction
  6) crop, remove floor, fill holes
  7) PointNet++ refinement (if model present)
  8) export OBJ + MTL + measurements.json + insole.stl bridge
"""
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


class ReconService:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ---------- public ----------

    def run_reconstruction(self, job_id: str, store):
        try:
            j = store.get(job_id)
            cloud_p = Path(j["cloud_path"])
            raw_dir = Path(j["raw_dir"])
            foot_side = j["foot_side"]

            pcd = self._load_ply(cloud_p)
            pcd = self._filter_confidence(pcd, min_conf=0.30)
            pcd = self._statistical_outlier(pcd, k=24, sigma=2.0)
            pcd = self._align_canonical(pcd, foot_side)

            # Try densify from raw depths if cloud sparse
            if len(pcd.points) < 8000:
                pcd = self._densify_from_depths(pcd, raw_dir, j)

            mesh = self._poisson(pcd, depth=9)
            mesh = self._crop_to_foot(mesh)
            mesh = self._fill_holes(mesh)
            mesh = self._smooth(mesh, iters=4)

            # Optional ML refinement
            mesh = self._pointnet_refine(mesh)

            out_dir = self.root / job_id / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            obj_p = out_dir / "foot.obj"
            self._write_obj(mesh, obj_p)

            store.update(job_id,
                         status="reconstructed",
                         mesh_path=str(obj_p))
        except Exception as e:
            store.update(job_id, status="failed", error=f"recon: {e}")

    def run_measure(self, job_id: str, store):
        try:
            j = store.get(job_id)
            mesh_p = Path(j["mesh_path"])
            mesh = o3d.io.read_triangle_mesh(str(mesh_p))
            mesh.compute_vertex_normals()
            m = self._measurements(mesh)

            out_dir = self.root / job_id / "out"
            mjson = out_dir / "measurements.json"
            mjson.write_text(json.dumps(m, indent=2))

            # bridge to existing insole builder (legacy)
            insole_p = out_dir / "insole.stl"
            try:
                from .legacy_bridge import build_insole_from_mesh
                build_insole_from_mesh(mesh_path=str(mesh_p),
                                       side=j["foot_side"],
                                       out_path=str(insole_p),
                                       measurements=m)
            except Exception:
                # fallback: simple offset insole
                self._fallback_insole(mesh, insole_p)

            store.update(job_id,
                         status="measured",
                         measurements_path=str(mjson),
                         insole_path=str(insole_p),
                         **{f"m_{k}": str(v) for k, v in m.items()})
        except Exception as e:
            store.update(job_id, status="failed", error=f"measure: {e}")

    # ---------- internals ----------

    def _load_ply(self, p: Path) -> o3d.geometry.PointCloud:
        # PLY written by Swift has 7 fields: x y z nx ny nz confidence
        # Read with custom parser to preserve confidence as separate array.
        lines = p.read_text().splitlines()
        hdr_end = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
        data = []
        for ln in lines[hdr_end + 1:]:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            data.append([float(x) for x in parts[:7]])
        arr = np.array(data, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[:, 0:3])
        pcd.normals = o3d.utility.Vector3dVector(arr[:, 3:6])
        # stash confidence in colors channel (R=conf)
        c = arr[:, 6:7].repeat(3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(c)
        return pcd

    def _filter_confidence(self, pcd, min_conf=0.30):
        if len(pcd.points) == 0:
            return pcd
        c = np.asarray(pcd.colors)[:, 0]
        keep = c >= min_conf
        if keep.sum() < 1000:
            return pcd
        return pcd.select_by_index(np.where(keep)[0])

    def _statistical_outlier(self, pcd, k=24, sigma=2.0):
        if len(pcd.points) < 200:
            return pcd
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=sigma)
        return pcd.select_by_index(ind)

    def _align_canonical(self, pcd, foot_side: str):
        """Centre at origin, rotate so longest axis = +X (toe direction)."""
        if len(pcd.points) < 100:
            return pcd
        pts = np.asarray(pcd.points)
        c = pts.mean(axis=0)
        pts = pts - c

        # PCA
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        R = evecs[:, order]   # cols: long, mid, short
        # ensure right-handed
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        # rotate so primary axis -> X
        pts = pts @ R

        # Flip so toe is +X. Heuristic: heel side has larger Z extent (sole flat).
        # For consistency, ensure +Y is medial side based on foot_side.
        if foot_side == "left":
            pts[:, 1] *= -1

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals) @ R
        )
        return pcd

    def _densify_from_depths(self, pcd, raw_dir: Path, j: dict):
        """Fuse raw depth maps into the cloud when initial cloud is sparse."""
        depth_dir = raw_dir / "depth"
        meta = json.loads(Path(j["meta_path"]).read_text())
        poses = meta.get("poses", [])
        if not depth_dir.exists() or not poses:
            return pcd

        extra_pts = []
        extra_nrm = []
        for pose in poses[::4]:  # sub-sample
            idx = pose.get("depth")
            if idx is None:
                continue
            bin_p = depth_dir / f"depth_{idx:05d}.bin"
            txt_p = depth_dir / f"depth_{idx:05d}.txt"
            if not bin_p.exists() or not txt_p.exists():
                continue
            w, h, _ = txt_p.read_text().split(",")
            w, h = int(w), int(h)
            d = np.fromfile(bin_p, dtype=np.float32).reshape(h, w)
            fx, fy, cx, cy = pose["intr"]
            anc = np.array(pose["anchor"], dtype=np.float32)

            ys, xs = np.where((d > 0.18) & (d < 0.55))
            if len(xs) == 0:
                continue
            zs = d[ys, xs]
            X = (xs.astype(np.float32) - cx) * zs / fx
            Y = (ys.astype(np.float32) - cy) * zs / fy
            P = np.stack([X, Y, zs], axis=1)
            # crop to anchor box
            L = P - anc
            mask = (np.abs(L) < np.array([0.16, 0.16, 0.10])).all(axis=1)
            P = P[mask]
            if len(P) > 0:
                extra_pts.append(P)
                # synthesise zero normals; recompute later
                extra_nrm.append(np.zeros_like(P))

        if not extra_pts:
            return pcd

        all_p = np.vstack([np.asarray(pcd.points)] + extra_pts)
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(all_p)
        merged = merged.voxel_down_sample(voxel_size=0.0025)
        merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.01, max_nn=24))
        merged.orient_normals_consistent_tangent_plane(20)
        return merged

    def _poisson(self, pcd, depth=9):
        if len(pcd.points) < 500:
            return o3d.geometry.TriangleMesh()
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=24))
            pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh \
            .create_from_point_cloud_poisson(pcd, depth=depth)
        if len(densities) == 0:
            return mesh
        d = np.asarray(densities)
        # drop low-density (extrapolated) faces
        keep = d > np.quantile(d, 0.05)
        verts_to_remove = np.where(~keep)[0]
        mesh.remove_vertices_by_index(verts_to_remove.tolist())
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _crop_to_foot(self, mesh):
        """Crop to plausible foot bounding box around origin."""
        if len(mesh.vertices) == 0:
            return mesh
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-0.18, -0.08, -0.06),
            max_bound=(0.18, 0.08, 0.10),
        )
        return mesh.crop(aabb)

    def _fill_holes(self, mesh):
        try:
            t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            t = t.fill_holes(hole_size=0.04)
            return t.to_legacy()
        except Exception:
            return mesh

    def _smooth(self, mesh, iters=4):
        if len(mesh.vertices) == 0:
            return mesh
        m = mesh.filter_smooth_taubin(number_of_iterations=iters)
        m.compute_vertex_normals()
        return m

    def _pointnet_refine(self, mesh):
        """Optional ML refinement. Loads weights if present, otherwise pass-through."""
        weights = Path("weights/pointnet_foot.pt")
        if not weights.exists() or len(mesh.vertices) < 1000:
            return mesh
        try:
            import torch
            from ..ml.pointnet_model import PointNetRefine
            net = PointNetRefine()
            net.load_state_dict(torch.load(str(weights), map_location="cpu"))
            net.eval()
            v = np.asarray(mesh.vertices, dtype=np.float32)
            # sample 4096 points, refine, broadcast residual back via KNN
            idx = np.random.choice(len(v), size=min(4096, len(v)), replace=False)
            sub = v[idx]
            with torch.no_grad():
                t = torch.from_numpy(sub).unsqueeze(0).transpose(1, 2)
                pred = net(t).squeeze(0).transpose(0, 1).numpy()
            delta = pred - sub
            # KNN broadcast
            tree = o3d.geometry.KDTreeFlann(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub)))
            new_v = v.copy()
            for i, p in enumerate(v):
                _, ids, _ = tree.search_knn_vector_3d(p, 1)
                new_v[i] = p + 0.5 * delta[ids[0]]
            mesh.vertices = o3d.utility.Vector3dVector(new_v)
            mesh.compute_vertex_normals()
        except Exception:
            pass
        return mesh

    def _write_obj(self, mesh, p: Path):
        if len(mesh.vertices) == 0:
            p.write_text("# empty\n")
            return
        o3d.io.write_triangle_mesh(str(p), mesh,
                                   write_vertex_normals=True,
                                   write_triangle_uvs=False)

    def _measurements(self, mesh) -> dict:
        v = np.asarray(mesh.vertices)
        if len(v) == 0:
            return {k: 0.0 for k in
                    ("length_mm", "ball_width_mm", "heel_width_mm",
                     "arch_height_mm", "instep_girth_mm")}
        # length along X
        length = (v[:, 0].max() - v[:, 0].min()) * 1000.0
        # ball width: slice 0.72L from heel, take Y extent
        xmin = v[:, 0].min()
        xspan = v[:, 0].max() - v[:, 0].min()
        ball_x = xmin + xspan * 0.72
        band = v[np.abs(v[:, 0] - ball_x) < xspan * 0.04]
        ball_w = ((band[:, 1].max() - band[:, 1].min()) * 1000.0
                  if len(band) > 5 else 0.0)
        # heel width at 0.10L
        heel_x = xmin + xspan * 0.10
        band2 = v[np.abs(v[:, 0] - heel_x) < xspan * 0.04]
        heel_w = ((band2[:, 1].max() - band2[:, 1].min()) * 1000.0
                  if len(band2) > 5 else 0.0)
        # arch height: max Z under midfoot (0.5L), measured above sole min
        mid = v[np.abs(v[:, 0] - (xmin + xspan * 0.50)) < xspan * 0.04]
        sole_min = v[:, 2].min()
        arch_h = ((mid[:, 2].max() - sole_min) * 1000.0
                  if len(mid) > 5 else 0.0)
        # instep girth: perimeter of slice at 0.55L (in Y-Z plane)
        ins = v[np.abs(v[:, 0] - (xmin + xspan * 0.55)) < xspan * 0.03]
        if len(ins) > 20:
            pts2 = ins[:, 1:3]
            c = pts2.mean(axis=0)
            d = pts2 - c
            ang = np.arctan2(d[:, 1], d[:, 0])
            order = np.argsort(ang)
            poly = pts2[order]
            seg = np.linalg.norm(np.diff(poly, axis=0, append=poly[:1]), axis=1)
            instep = seg.sum() * 1000.0
        else:
            instep = 0.0
        return {
            "length_mm": round(float(length), 1),
            "ball_width_mm": round(float(ball_w), 1),
            "heel_width_mm": round(float(heel_w), 1),
            "arch_height_mm": round(float(arch_h), 1),
            "instep_girth_mm": round(float(instep), 1),
        }

    def _fallback_insole(self, foot_mesh, out_path: Path):
        """Crude offset insole if legacy builder unavailable."""
        if len(foot_mesh.vertices) == 0:
            out_path.write_bytes(b"")
            return
        # Take bottom 30% of foot by Z, mirror upward, smooth
        v = np.asarray(foot_mesh.vertices)
        sole_thr = np.quantile(v[:, 2], 0.30)
        keep = v[:, 2] <= sole_thr
        sub = foot_mesh.select_by_index(np.where(keep)[0])
        if len(sub.vertices) < 100:
            o3d.io.write_triangle_mesh(str(out_path), foot_mesh)
            return
        v2 = np.asarray(sub.vertices)
        v2[:, 2] = sole_thr * 2 - v2[:, 2]
        sub.vertices = o3d.utility.Vector3dVector(v2)
        sub.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_path), sub)
