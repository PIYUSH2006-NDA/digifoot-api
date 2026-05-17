"""
backend/ml_training/data/synthetic_gen.py
Generate paired (noisy_scan, clean_mesh) training data.

Requires: foot_template.ply in --template path.

Usage:
  python -m ml_training.data.synthetic_gen \
    --out data/foot_dataset/synthetic \
    --template assets/foot_template.ply \
    --n 10000 --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# parametric foot shape ranges (in mm)
PARAMS = {
    "length":      (220.0, 310.0),
    "ball_width":  (82.0,  118.0),
    "heel_width":  (52.0,  78.0),
    "arch_height": (8.0,   38.0),
    "toe_spread":  (0.82,  1.18),
    "instep":      (215.0, 290.0),
    "hallux_va":   (-5.0,  28.0),     # degrees
}


def _sample_params(rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAMS.items()}


def _morph_template(template: o3d.geometry.TriangleMesh,
                    p: dict, rng: np.random.Generator
                    ) -> o3d.geometry.TriangleMesh:
    v = np.asarray(template.vertices).copy()

    # global scale to target length
    bbox_len = max(v[:, 0].ptp(), 1e-3)
    scale_x = p["length"] / bbox_len
    scale_y = p["ball_width"] / max(v[:, 1].ptp(), 1e-3)
    v[:, 0] *= scale_x
    v[:, 1] *= scale_y

    # arch lift: gaussian bump under midfoot
    mid = (v[:, 0].min() + v[:, 0].max()) / 2.0
    sigma = 0.15 * v[:, 0].ptp()
    bump = p["arch_height"] * np.exp(-((v[:, 0] - mid) ** 2) / (2 * sigma ** 2))
    sole_mask = v[:, 2] < (v[:, 2].min() + 0.1 * v[:, 2].ptp())
    v[sole_mask, 2] += bump[sole_mask]

    # heel narrowing
    heel_x = v[:, 0].min()
    heel_mask = v[:, 0] < heel_x + 0.25 * v[:, 0].ptp()
    heel_spread = max(v[heel_mask, 1].ptp(), 1e-3)
    v[heel_mask, 1] *= p["heel_width"] / heel_spread

    # toe spread
    toe_mask = v[:, 0] > v[:, 0].min() + 0.85 * v[:, 0].ptp()
    v[toe_mask, 1] *= p["toe_spread"]

    # hallux valgus rotation
    angle = np.deg2rad(p["hallux_va"])
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]], dtype=np.float64)
    big_toe = (v[:, 0] > v[:, 0].min() + 0.92 * v[:, 0].ptp()) & (v[:, 1] > 0)
    if big_toe.any():
        c = v[big_toe].mean(0)
        v[big_toe] = (v[big_toe] - c) @ R.T + c

    # micro noise for skin individuality
    v += rng.normal(0, 0.3, v.shape)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(v)
    out.triangles = template.triangles
    out.compute_vertex_normals()
    return out


def _simulate_scan(clean: o3d.geometry.TriangleMesh,
                   rng: np.random.Generator,
                   n_pts: int = 50_000
                   ) -> np.ndarray:
    """Simulate noisy TrueDepth scan from clean mesh."""
    pts = np.asarray(clean.sample_points_poisson_disk(n_pts).points)

    # axial noise: ~1.2mm σ
    axial = rng.normal(0, 1.2, (len(pts), 1))
    # lateral noise: ~0.8mm σ
    lateral = rng.normal(0, 0.8, (len(pts), 2))
    pts_noisy = pts + np.concatenate([lateral, axial], axis=-1)

    # drop 5-15% random patches (simulate occlusion / missed coverage)
    drop_frac = rng.uniform(0.05, 0.15)
    n_keep = int(len(pts_noisy) * (1.0 - drop_frac))
    keep_idx = rng.choice(len(pts_noisy), n_keep, replace=False)
    pts_noisy = pts_noisy[keep_idx]

    # add 1% scattered outliers
    n_out = max(1, int(len(pts_noisy) * 0.01))
    outlier_base = pts_noisy[rng.choice(len(pts_noisy), n_out)]
    outliers = outlier_base + rng.normal(0, 8.0, (n_out, 3))
    pts_noisy = np.concatenate([pts_noisy, outliers], axis=0)

    return pts_noisy.astype(np.float32)


def generate(out_dir: Path, n: int, template_path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = o3d.io.read_triangle_mesh(str(template_path))
    if len(template.vertices) == 0:
        raise ValueError(f"empty template: {template_path}")
    template.compute_vertex_normals()
    print(f"[synth] template: {len(template.vertices)} verts")
    print(f"[synth] generating {n} samples → {out_dir}")

    for i in range(n):
        p = _sample_params(rng)
        clean = _morph_template(template, p, rng)
        scan = _simulate_scan(clean, rng)

        np.save(out_dir / f"{i:06d}_scan.npy", scan)
        o3d.io.write_triangle_mesh(
            str(out_dir / f"{i:06d}_clean.ply"), clean,
            write_ascii=False)
        (out_dir / f"{i:06d}_params.json").write_text(json.dumps(p))

        if i % 500 == 0:
            print(f"  [{i}/{n}] scan pts={len(scan)}, "
                  f"clean verts={len(clean.vertices)}")

    print(f"[synth] done. {n} pairs at {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True,
                    help="path to base foot mesh .ply/.obj")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(Path(a.out), a.n, Path(a.template), a.seed)
