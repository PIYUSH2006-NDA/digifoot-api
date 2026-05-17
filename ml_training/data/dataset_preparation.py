# dataset_preparation.py
# TrueDepth frames → YOLO segmentation training data
# Handles: preprocessing, auto-masking, augmentation, synthetic generation

import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


# ======================================================================
#  DATASET BUILDER
# ======================================================================

class DepthDatasetBuilder:
    """
    Convert raw TrueDepth depth maps → YOLO segmentation training dataset.

    Folder structure produced:
        output_dir/
          dataset.yaml
          train/images/*.png
          train/labels/*.txt
          val/images/*.png
          val/labels/*.txt
          test/images/*.png
          test/labels/*.txt
    """

    def __init__(self, output_dir: str, preprocessor, segmenter):
        self.out = Path(output_dir)
        self.prep = preprocessor
        self.seg = segmenter
        self._setup_dirs()

    def _setup_dirs(self):
        for split in ["train", "val", "test"]:
            (self.out / split / "images").mkdir(parents=True, exist_ok=True)
            (self.out / split / "labels").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  YOLO label generation
    # ------------------------------------------------------------------ #

    def mask_to_yolo_polygon(
        self,
        mask: np.ndarray,
        img_w: int,
        img_h: int,
        simplify_epsilon_factor: float = 0.005,
    ) -> List[float]:
        """
        Binary mask → YOLO segmentation polygon (normalized [0,1] coords).
        Format: [x1, y1, x2, y2, ..., xn, yn]
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        cnt = max(contours, key=cv2.contourArea)
        epsilon = simplify_epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            return []

        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        pts = np.clip(pts, 0.0, 1.0)
        return pts.flatten().tolist()

    def write_yolo_label(
        self,
        label_path: Path,
        polygon: List[float],
        class_id: int = 0,
    ):
        with open(label_path, "w") as f:
            if polygon:
                coords = " ".join(f"{v:.6f}" for v in polygon)
                f.write(f"{class_id} {coords}\n")

    # ------------------------------------------------------------------ #
    #  Sample processing
    # ------------------------------------------------------------------ #

    def process_sample(
        self,
        depth_input,
        split: str = "train",
        idx: int = 0,
        manual_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Process one depth frame → YOLO training sample.

        Args:
            depth_input: file path (str) or raw uint16 ndarray
            split:       'train' | 'val' | 'test'
            idx:         sample index for filename
            manual_mask: optional hand-labeled mask (overrides auto-seg)
        """
        processed = self.prep.preprocess_full(depth_input)
        depth_m = processed["raw_meters"]

        if manual_mask is not None:
            mask = manual_mask
            valid = True
        else:
            result = self.seg.isolate_foot(depth_m)
            mask = result["mask"]
            valid = result["valid"]

        if not valid or mask is None or mask.sum() < 500:
            print(f"  [SKIP] idx={idx}: no valid foot mask")
            return False

        img = processed["pseudo_rgb"]
        h, w = img.shape[:2]
        polygon = self.mask_to_yolo_polygon(mask, w, h)

        base = f"foot_{idx:05d}"
        cv2.imwrite(str(self.out / split / "images" / f"{base}.png"), img)
        self.write_yolo_label(self.out / split / "labels" / f"{base}.txt", polygon)
        return True

    def process_directory(
        self,
        depth_dir: str,
        split: str = "train",
        start_idx: int = 0,
    ) -> int:
        """Process all .png depth maps in a directory."""
        depth_dir = Path(depth_dir)
        depth_files = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.tiff"))
        success = 0
        for i, path in enumerate(depth_files):
            ok = self.process_sample(str(path), split=split, idx=start_idx + i)
            if ok:
                success += 1
        print(f"Processed {success}/{len(depth_files)} from {depth_dir}")
        return success

    # ------------------------------------------------------------------ #
    #  Dataset YAML
    # ------------------------------------------------------------------ #

    def write_dataset_yaml(self, nc: int = 1, names: List[str] = None):
        names = names or ["foot"]
        yaml = (
            f"path: {self.out.absolute()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n\n"
            f"nc: {nc}\n"
            f"names: {names}\n"
        )
        with open(self.out / "dataset.yaml", "w") as f:
            f.write(yaml)
        print(f"Wrote: {self.out / 'dataset.yaml'}")

    def print_stats(self):
        """Print sample counts per split."""
        for split in ["train", "val", "test"]:
            imgs = list((self.out / split / "images").glob("*.png"))
            lbls = list((self.out / split / "labels").glob("*.txt"))
            print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")


# ======================================================================
#  AUGMENTOR
# ======================================================================

class DepthAugmentor:
    """
    Depth-specific augmentation.
    Generates diverse training samples from limited real captures.
    """

    def augment_pair(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate augmented (depth, mask) pairs from one real sample.
        Typical expansion: 1 → ~30 samples.
        """
        results = [(depth.copy(), mask.copy())]  # original

        # 1. Horizontal flip (generates opposite foot effectively)
        results.append((np.fliplr(depth), np.fliplr(mask)))

        # 2. Rotation
        for angle in [-20, -15, -10, -5, 5, 10, 15, 20]:
            d, m = self._rotate(depth, mask, angle)
            results.append((d, m))

        # 3. Depth-value noise (sensor variation)
        for sigma in [0.002, 0.004, 0.007]:
            results.append((self._add_depth_noise(depth, sigma), mask.copy()))

        # 4. Distance scale (closer / farther scan)
        for scale in [0.80, 0.90, 1.10, 1.20]:
            results.append((depth * scale, mask.copy()))

        # 5. Sensor dropout (random holes)
        for hole_ratio in [0.03, 0.07]:
            results.append((self._add_holes(depth, hole_ratio), mask.copy()))

        # 6. Translation
        for tx, ty in [(-25, 0), (25, 0), (0, -25), (0, 25)]:
            d, m = self._translate(depth, mask, tx, ty)
            results.append((d, m))

        # 7. Blur variation (different smoothing conditions)
        for ksize in [3, 5]:
            blurred = cv2.GaussianBlur(depth, (ksize, ksize), 0)
            results.append((blurred, mask.copy()))

        return results

    def _rotate(self, depth, mask, angle):
        h, w = depth.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        d = cv2.warpAffine(
            depth, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        m = cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return d, m

    def _translate(self, depth, mask, tx, ty):
        h, w = depth.shape
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        d = cv2.warpAffine(depth, M, (w, h))
        m = cv2.warpAffine(mask, M, (w, h))
        return d, m

    def _add_depth_noise(self, depth: np.ndarray, sigma: float) -> np.ndarray:
        noise = np.random.normal(0, sigma, depth.shape).astype(np.float32)
        return np.clip(depth + noise, 0, None)

    def _add_holes(self, depth: np.ndarray, ratio: float = 0.05) -> np.ndarray:
        result = depth.copy()
        n = int(depth.size * ratio)
        idx = np.random.choice(depth.size, n, replace=False)
        result.flat[idx] = 0.0
        return result


# ======================================================================
#  SYNTHETIC DEPTH GENERATION
# ======================================================================

def generate_synthetic_foot(
    h: int = 480,
    w: int = 640,
    base_depth: float = 0.60,
    floor_depth: float = 0.65,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic foot-shaped depth map for pre-training.
    Not anatomically precise — teaches model depth blob geometry.

    Returns:
        depth: float32 depth map (meters)
        mask:  uint8 binary mask (0 / 255)
    """
    if random_state is not None:
        np.random.seed(random_state)

    depth = np.full((h, w), floor_depth, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    cx = int(w * np.random.uniform(0.35, 0.65))
    cy = int(h * np.random.uniform(0.35, 0.65))
    foot_angle = np.random.uniform(-35, 35)

    foot_w = int(np.random.uniform(50, 80))
    foot_h = int(np.random.uniform(130, 200))

    # Main foot ellipse
    cv2.ellipse(depth, (cx, cy), (foot_w, foot_h), foot_angle, 0, 360, base_depth, -1)
    cv2.ellipse(mask, (cx, cy), (foot_w, foot_h), foot_angle, 0, 360, 255, -1)

    # Heel (slightly wider ellipse at one end)
    heel_offset_x = int(foot_h * 0.55 * np.sin(np.radians(foot_angle)))
    heel_offset_y = int(foot_h * 0.55 * np.cos(np.radians(foot_angle)))
    cv2.ellipse(
        depth,
        (cx + heel_offset_x, cy + heel_offset_y),
        (int(foot_w * 1.1), int(foot_w * 0.9)),
        foot_angle, 0, 360, base_depth + 0.005, -1,
    )
    cv2.ellipse(
        mask,
        (cx + heel_offset_x, cy + heel_offset_y),
        (int(foot_w * 1.1), int(foot_w * 0.9)),
        foot_angle, 0, 360, 255, -1,
    )

    # Toes (5 small circles at front end)
    front_x = cx - int(foot_h * 0.55 * np.sin(np.radians(foot_angle)))
    front_y = cy - int(foot_h * 0.55 * np.cos(np.radians(foot_angle)))
    perp_angle = np.radians(foot_angle + 90)

    toe_sizes = [14, 12, 11, 10, 9]
    toe_spacing = foot_w * 2 / 5
    for i in range(5):
        offset = (i - 2) * toe_spacing
        tx = int(front_x + offset * np.cos(perp_angle))
        ty = int(front_y + offset * np.sin(perp_angle))
        r = toe_sizes[i] + np.random.randint(-2, 3)
        cv2.circle(depth, (tx, ty), r, base_depth - 0.008, -1)
        cv2.circle(mask, (tx, ty), r, 255, -1)

    # Arch cavity (higher depth = farther from camera = recessed)
    arch_x = cx + int(foot_h * 0.15 * np.sin(np.radians(foot_angle)))
    arch_y = cy + int(foot_h * 0.15 * np.cos(np.radians(foot_angle)))
    arch_mask_local = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        arch_mask_local,
        (arch_x, arch_y),
        (int(foot_w * 0.35), int(foot_h * 0.22)),
        foot_angle, 0, 360, 1, -1,
    )
    depth += arch_mask_local * 0.018

    # Gaussian depth noise (sensor simulation)
    noise = np.random.normal(0, 0.003, (h, w)).astype(np.float32)
    depth += noise

    # Random dropout holes
    hole_mask = np.random.random((h, w)) < 0.03
    depth[hole_mask & (mask > 0)] = 0.0

    return depth, mask


def generate_synthetic_dataset(
    output_dir: str,
    n_samples: int = 500,
    train_frac: float = 0.80,
    val_frac: float = 0.15,
):
    """Generate full synthetic dataset for pre-training."""
    from depth_preprocessing import DepthPreprocessor
    builder_prep = DepthPreprocessor()

    out = Path(output_dir)
    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    splits = []
    for i in range(n_samples):
        if i < int(n_samples * train_frac):
            splits.append("train")
        elif i < int(n_samples * (train_frac + val_frac)):
            splits.append("val")
        else:
            splits.append("test")

    for i, split in enumerate(splits):
        depth, mask = generate_synthetic_foot(random_state=i)
        img = builder_prep.to_pseudo_rgb(depth)
        h, w = img.shape[:2]

        base = f"synth_{i:05d}"
        cv2.imwrite(str(out / split / "images" / f"{base}.png"), img)

        # Polygon label
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label = ""
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                pts = approx.reshape(-1, 2).astype(float)
                pts[:, 0] /= w
                pts[:, 1] /= h
                pts = np.clip(pts, 0, 1)
                coords = " ".join(f"{v:.6f}" for v in pts.flatten())
                label = f"0 {coords}\n"

        with open(out / split / "labels" / f"{base}.txt", "w") as f:
            f.write(label)

        if i % 50 == 0:
            print(f"  Generated {i+1}/{n_samples}")

    # Write YAML
    yaml = (
        f"path: {out.absolute()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n\n"
        "nc: 1\nnames: ['foot']\n"
    )
    with open(out / "dataset.yaml", "w") as f:
        f.write(yaml)

    print(f"\nSynthetic dataset: {n_samples} samples → {out}")


# ======================================================================
#  CLI
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foot scan dataset tools")
    sub = parser.add_subparsers(dest="cmd")

    p_synth = sub.add_parser("synthetic", help="Generate synthetic dataset")
    p_synth.add_argument("--output", default="synthetic_dataset")
    p_synth.add_argument("--n", type=int, default=500)

    args = parser.parse_args()

    if args.cmd == "synthetic":
        generate_synthetic_dataset(args.output, n_samples=args.n)
    else:
        parser.print_help()
