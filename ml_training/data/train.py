"""
backend/ml_training/data/train.py
Train PointNet2FootRefine.

Losses:
  L = chamfer(pred, gt) + λ_n * normal_consistency + λ_s * smooth_reg

Usage:
  python -m ml_training.data.train \
    --data data/foot_dataset \
    --out weights/pointnet2_foot_v1.pt \
    --epochs 150 --bs 8 --lr 1e-3
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# model lives at backend/app/ml/pointnet2_model.py
# add backend/ to sys.path via train_all.sh or PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.ml.pointnet2_model import PointNet2FootRefine

from .dataset import FootDataset

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import knn_points, knn_gather
    _P3D = True
except ImportError:
    _P3D = False


# ─────────── losses ───────────

def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [B, N, 3]"""
    if _P3D:
        d, _ = chamfer_distance(pred, gt)
        return d
    d = torch.cdist(pred, gt)
    return d.min(-1).values.mean() + d.min(-2).values.mean()


def normal_loss(pred_nrm: torch.Tensor, pred_pts: torch.Tensor,
                gt_pts: torch.Tensor, gt_nrm: torch.Tensor) -> torch.Tensor:
    """Match each predicted normal to nearest GT normal. Higher = worse."""
    if not _P3D:
        return torch.tensor(0.0, device=pred_nrm.device)
    idx = knn_points(pred_pts, gt_pts, K=1).idx        # [B,N,1]
    nn_nrm = knn_gather(gt_nrm, idx).squeeze(2)        # [B,N,3]
    cos = (pred_nrm * nn_nrm).sum(-1).abs()             # [B,N]
    return (1.0 - cos).mean()


def smooth_loss(pts: torch.Tensor) -> torch.Tensor:
    """Encourage local uniformity via kNN distance variance."""
    if not _P3D:
        return torch.tensor(0.0, device=pts.device)
    d = knn_points(pts, pts, K=8).dists                 # [B,N,8]
    return d.std(-1).mean()


# ─────────── metrics ───────────

def f_score(pred: torch.Tensor, gt: torch.Tensor,
            threshold: float = 0.003) -> torch.Tensor:
    d = torch.cdist(pred, gt)
    p = (d.min(-1).values < threshold).float().mean(-1)
    r = (d.min(-2).values < threshold).float().mean(-1)
    return (2 * p * r / (p + r + 1e-8)).mean()


# ─────────── train ───────────

def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={dev}")

    train_ds = FootDataset(args.data, n_pts=args.npts, split="train")
    val_ds   = FootDataset(args.data, n_pts=args.npts, split="val")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, num_workers=2)
    print(f"[train] train={len(train_ds)}, val={len(val_ds)}")

    model = PointNet2FootRefine().to(dev)
    if args.resume and Path(args.resume).exists():
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"[train] resumed from {args.resume}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best_cd = 1e9

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        n_batch = 0

        for batch in train_dl:
            scan   = batch["scan"].to(dev)
            gt_pts = batch["gt_pts"].to(dev)
            gt_nrm = batch["gt_nrm"].to(dev)

            disp, nrm_pred = model(scan)
            pred_pts = scan + disp

            lc = chamfer_loss(pred_pts, gt_pts)
            ln = normal_loss(nrm_pred, pred_pts, gt_pts, gt_nrm)
            ls = smooth_loss(pred_pts)
            loss = lc + 0.5 * ln + 0.1 * ls

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            n_batch += 1

        sched.step()

        # ── validation ──
        model.eval()
        val_cd = 0.0
        val_fs = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                scan   = batch["scan"].to(dev)
                gt_pts = batch["gt_pts"].to(dev)
                disp, _ = model(scan)
                pred = scan + disp
                val_cd += chamfer_loss(pred, gt_pts).item()
                val_fs += f_score(pred, gt_pts, 0.003).item()
                val_n += 1

        val_cd /= max(val_n, 1)
        val_fs /= max(val_n, 1)
        dt = time.time() - t0

        print(f"ep {ep:03d} | train_loss {ep_loss/n_batch:.5f} | "
              f"val_cd {val_cd:.6f} | F@3mm {val_fs:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {dt:.1f}s")

        # ── save best ──
        if val_cd < best_cd:
            best_cd = val_cd
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ saved → {args.out} (cd={val_cd:.6f})")

    print(f"[train] done. best val_cd={best_cd:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True,
                    help="root dir with synthetic/ and optional organic/")
    ap.add_argument("--out",    required=True,
                    help="output weights path, e.g. weights/pointnet2_foot_v1.pt")
    ap.add_argument("--epochs", type=int,   default=150)
    ap.add_argument("--bs",     type=int,   default=8)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--npts",   type=int,   default=16_384)
    ap.add_argument("--resume", type=str,   default=None)
    train(ap.parse_args())
