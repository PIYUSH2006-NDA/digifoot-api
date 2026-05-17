"""
backend/ml_training/data/eval.py
Evaluate trained PointNet2FootRefine on test split.

Reports: chamfer (normalized), F@1mm, F@2mm, F@3mm, normal cosine.

Usage:
  python -m ml_training.data.eval \
    --data data/foot_dataset \
    --weights weights/pointnet2_foot_v1.pt \
    --npts 16384
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.ml.pointnet2_model import PointNet2FootRefine

from .dataset import FootDataset

try:
    from pytorch3d.ops import knn_points, knn_gather
    _P3D = True
except ImportError:
    _P3D = False


def chamfer(pred, gt):
    d = torch.cdist(pred, gt)
    return d.min(-1).values.mean() + d.min(-2).values.mean()


def f_score(pred, gt, thr):
    d = torch.cdist(pred, gt)
    p = (d.min(-1).values < thr).float().mean(-1)
    r = (d.min(-2).values < thr).float().mean(-1)
    return (2 * p * r / (p + r + 1e-8)).mean()


def normal_cos(pred_nrm, pred_pts, gt_pts, gt_nrm):
    if not _P3D:
        return torch.tensor(0.0)
    idx = knn_points(pred_pts, gt_pts, K=1).idx
    nn_nrm = knn_gather(gt_nrm, idx).squeeze(2)
    return (pred_nrm * nn_nrm).sum(-1).abs().mean()


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FootDataset(a.data, n_pts=a.npts, split="test")
    dl = DataLoader(ds, batch_size=4, num_workers=2)
    print(f"[eval] test samples={len(ds)}, device={dev}")

    model = PointNet2FootRefine().to(dev).eval()
    model.load_state_dict(torch.load(a.weights, map_location=dev))

    cd_all = []
    f1_all = []
    f2_all = []
    f3_all = []
    nc_all = []

    with torch.no_grad():
        for batch in dl:
            scan   = batch["scan"].to(dev)
            gt_pts = batch["gt_pts"].to(dev)
            gt_nrm = batch["gt_nrm"].to(dev)

            disp, nrm = model(scan)
            pred = scan + disp

            cd_all.append(chamfer(pred, gt_pts).item())
            f1_all.append(f_score(pred, gt_pts, 0.001).item())
            f2_all.append(f_score(pred, gt_pts, 0.002).item())
            f3_all.append(f_score(pred, gt_pts, 0.003).item())
            nc_all.append(normal_cos(nrm, pred, gt_pts, gt_nrm).item())

    results = {
        "n_samples":     len(ds),
        "chamfer_norm":  float(np.mean(cd_all)),
        "f_at_1mm":      float(np.mean(f1_all)),
        "f_at_2mm":      float(np.mean(f2_all)),
        "f_at_3mm":      float(np.mean(f3_all)),
        "normal_cos":    float(np.mean(nc_all)),
    }

    print(json.dumps(results, indent=2))

    out = Path(a.weights).parent / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved → {out}")

    # pass/fail
    print("\n── pass criteria ──")
    print(f"  F@1mm ≥ 0.85:  {'PASS' if results['f_at_1mm'] >= 0.85 else 'FAIL'} ({results['f_at_1mm']:.4f})")
    print(f"  F@2mm ≥ 0.97:  {'PASS' if results['f_at_2mm'] >= 0.97 else 'FAIL'} ({results['f_at_2mm']:.4f})")
    print(f"  chamfer < 0.001: {'PASS' if results['chamfer_norm'] < 0.001 else 'FAIL'} ({results['chamfer_norm']:.6f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--npts",    type=int, default=16_384)
    main(ap.parse_args())
