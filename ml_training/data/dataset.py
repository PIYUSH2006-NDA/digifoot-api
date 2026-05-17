"""
backend/ml_training/data/dataset.py
Foot scan dataset. Mixes synthetic + organic (when available).
"""
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from pathlib import Path


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    c = pts.mean(0)
    s = float(np.linalg.norm(pts - c, axis=1).max() + 1e-8)
    return (pts - c) / s, c, s


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """FPS downsampling — better coverage than random."""
    idx = np.zeros(n, dtype=np.int64)
    dists = np.full(len(pts), np.inf, dtype=np.float64)
    farthest = np.random.randint(len(pts))
    for i in range(n):
        idx[i] = farthest
        diff = pts - pts[farthest]
        d = (diff * diff).sum(-1)
        dists = np.minimum(dists, d)
        farthest = int(dists.argmax())
    return idx


def _rand_rotation() -> np.ndarray:
    """Random Z-axis rotation."""
    a = np.random.uniform(-np.pi, np.pi)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


class FootDataset(Dataset):
    """
    Expects:
      root/synthetic/000000_scan.npy, 000000_clean.ply, ...
      root/organic/000000_scan.npy,   000000_clean.ply, ...  (optional)

    Returns per sample:
      scan:    [N, 3] noisy input points (normalized)
      gt_pts:  [N, 3] clean target points (same normalization)
      gt_nrm:  [N, 3] clean target normals
    """
    def __init__(self, root: str | Path, n_pts: int = 16_384,
                 organic_ratio: float = 0.3, split: str = "train"):
        root = Path(root)
        self.n = n_pts
        self.split = split

        syn = sorted((root / "synthetic").glob("*_scan.npy"))
        org_dir = root / "organic"
        org = sorted(org_dir.glob("*_scan.npy")) if org_dir.exists() else []

        self.syn = self._split_list(syn)
        self.org = self._split_list(org) if org else []
        self.organic_ratio = organic_ratio if self.org else 0.0

    def _split_list(self, lst: list) -> list:
        k1 = int(len(lst) * 0.8)
        k2 = int(len(lst) * 0.9)
        return {"train": lst[:k1], "val": lst[k1:k2], "test": lst[k2:]}[self.split]

    def __len__(self) -> int:
        if self.org:
            return int(len(self.syn) / (1.0 - self.organic_ratio))
        return len(self.syn)

    def _pick_path(self, idx: int) -> Path:
        if self.org and np.random.rand() < self.organic_ratio:
            return self.org[idx % len(self.org)]
        return self.syn[idx % len(self.syn)]

    def __getitem__(self, idx: int) -> dict:
        scan_path = self._pick_path(idx)
        clean_path = scan_path.with_name(
            scan_path.name.replace("_scan.npy", "_clean.ply"))

        # load
        scan = np.load(scan_path).astype(np.float32)
        clean_mesh = o3d.io.read_triangle_mesh(str(clean_path))
        clean_mesh.compute_vertex_normals()
        clean_pcd = clean_mesh.sample_points_poisson_disk(self.n)
        clean_pts = np.asarray(clean_pcd.points, dtype=np.float32)
        clean_nrm = np.asarray(clean_pcd.normals, dtype=np.float32)

        # downsample scan to n via FPS
        if len(scan) > self.n:
            sel = _farthest_point_sample(scan, self.n)
            scan = scan[sel]
        elif len(scan) < self.n:
            pad = np.random.choice(len(scan), self.n - len(scan))
            scan = np.concatenate([scan, scan[pad]], axis=0)

        # shared normalization
        scan_n, c, s = _normalize(scan)
        clean_n = (clean_pts - c) / s

        # augment (train only)
        if self.split == "train":
            R = _rand_rotation()
            scan_n = scan_n @ R.T
            clean_n = clean_n @ R.T
            clean_nrm = clean_nrm @ R.T
            # random jitter
            scan_n += np.random.normal(0, 0.002, scan_n.shape).astype(np.float32)

        return {
            "scan":   torch.from_numpy(scan_n),
            "gt_pts": torch.from_numpy(clean_n),
            "gt_nrm": torch.from_numpy(clean_nrm),
        }
