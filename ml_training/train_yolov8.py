# train_yolov8.py
# YOLOv8-seg training for depth-only foot segmentation
# Optimized for tiny datasets (30–500 samples)

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


# ======================================================================
#  STAGE 1 — Frozen backbone (transfer learning)
# ======================================================================

def train_stage1(
    data_yaml: str = "foot_dataset/dataset.yaml",
    model_variant: str = "yolov8n-seg",   # nano: best for CoreML / mobile
    epochs: int = 150,
    img_size: int = 640,
    batch: int = 8,
    device: str = "0",                    # GPU index or "cpu" or "mps"
    output_name: str = "foot_seg_stage1",
) -> str:
    """
    Stage 1: Freeze backbone, train head only.
    Prevents overfitting on tiny datasets.
    Use yolov8n-seg for mobile, yolov8s-seg for better accuracy.
    """
    model = YOLO(f"{model_variant}.pt")  # COCO pretrained weights

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,

        # ── Optimizer ──────────────────────────────────────────────── #
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        # ── Augmentation (depth-tuned) ─────────────────────────────── #
        # Disable color-space augments (depth has no color)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,       # tiny brightness variation OK for pseudo-RGB

        # Geometric augments (all valid for depth)
        degrees=20.0,
        translate=0.10,
        scale=0.25,
        shear=5.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,       # left/right foot flip

        # Mosaic/mixup: helpful for small datasets
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.1,

        # ── Segmentation ──────────────────────────────────────────── #
        overlap_mask=True,
        mask_ratio=4,

        # ── Transfer learning: freeze backbone ─────────────────────── #
        freeze=10,        # freeze first 10 layers (full backbone)

        # ── Regularization ─────────────────────────────────────────── #
        label_smoothing=0.10,
        dropout=0.0,      # YOLO handles this internally

        # ── Training control ───────────────────────────────────────── #
        patience=50,
        save=True,
        save_period=25,
        workers=4,
        project="foot_scan_runs",
        name=output_name,
        exist_ok=True,
        verbose=True,
    )

    best_path = str(Path("foot_scan_runs") / output_name / "weights" / "best.pt")
    print(f"\nStage 1 complete. Best weights: {best_path}")
    return best_path


# ======================================================================
#  STAGE 2 — Full fine-tuning
# ======================================================================

def train_stage2(
    stage1_weights: str,
    data_yaml: str = "foot_dataset/dataset.yaml",
    epochs: int = 100,
    batch: int = 4,
    device: str = "0",
    output_name: str = "foot_seg_stage2",
) -> str:
    """
    Stage 2: Unfreeze all layers, fine-tune with low LR.
    Run after Stage 1 converges.
    """
    model = YOLO(stage1_weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,

        optimizer="AdamW",
        lr0=0.0001,          # very low LR for fine-tuning
        lrf=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,

        freeze=0,            # unfreeze everything

        # Same augments as stage 1
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.05,
        degrees=15.0, translate=0.08, scale=0.20,
        fliplr=0.5, mosaic=0.5, mixup=0.05,
        overlap_mask=True, mask_ratio=4,
        label_smoothing=0.05,

        patience=30,
        save=True,
        save_period=20,
        project="foot_scan_runs",
        name=output_name,
        exist_ok=True,
    )

    best_path = str(Path("foot_scan_runs") / output_name / "weights" / "best.pt")
    print(f"\nStage 2 complete. Best weights: {best_path}")
    return best_path


# ======================================================================
#  EXPORT
# ======================================================================

def export_coreml(model_path: str, img_size: int = 640):
    """Export to CoreML (.mlpackage) for iOS deployment."""
    model = YOLO(model_path)
    model.export(
        format="coreml",
        imgsz=img_size,
        half=True,        # FP16 — uses Apple Neural Engine
        nms=True,         # embed NMS in graph
        simplify=True,
        opset=16,
    )
    print(f"CoreML exported: {model_path.replace('.pt', '.mlpackage')}")


def export_onnx(model_path: str, img_size: int = 640):
    """Export to ONNX with FP16 quantization."""
    model = YOLO(model_path)
    model.export(
        format="onnx",
        imgsz=img_size,
        simplify=True,
        opset=16,
        dynamic=False,
        half=True,
    )
    print(f"ONNX exported: {model_path.replace('.pt', '.onnx')}")


def export_tflite(model_path: str, img_size: int = 640):
    """Export to TFLite INT8 (Android / embedded deployment)."""
    model = YOLO(model_path)
    model.export(
        format="tflite",
        imgsz=img_size,
        int8=True,
    )


# ======================================================================
#  VALIDATION
# ======================================================================

def validate(model_path: str, data_yaml: str):
    """Run validation and print metrics."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=640, verbose=True)
    print(f"\nBox mAP50:     {metrics.box.map50:.4f}")
    print(f"Box mAP50-95:  {metrics.box.map:.4f}")
    print(f"Seg mAP50:     {metrics.seg.map50:.4f}")
    print(f"Seg mAP50-95:  {metrics.seg.map:.4f}")
    return metrics


# ======================================================================
#  TWO-PHASE TRAINING PIPELINE
# ======================================================================

def full_training_pipeline(
    data_yaml: str,
    model_variant: str = "yolov8n-seg",
    device: str = "0",
    export_formats: list = ["coreml", "onnx"],
):
    """Run complete two-stage training + export."""
    print("=" * 60)
    print("STAGE 1: Frozen backbone training")
    print("=" * 60)
    s1_weights = train_stage1(
        data_yaml=data_yaml,
        model_variant=model_variant,
        device=device,
    )

    print("\n" + "=" * 60)
    print("STAGE 2: Full fine-tuning")
    print("=" * 60)
    s2_weights = train_stage2(
        stage1_weights=s1_weights,
        data_yaml=data_yaml,
        device=device,
    )

    print("\n" + "=" * 60)
    print("EXPORT")
    print("=" * 60)
    if "coreml" in export_formats:
        export_coreml(s2_weights)
    if "onnx" in export_formats:
        export_onnx(s2_weights)

    return s2_weights


# ======================================================================
#  HYPERPARAMETER SEARCH (optional)
# ======================================================================

def hyperparameter_tune(data_yaml: str, n_trials: int = 10):
    """
    YOLOv8 built-in hyperparameter evolution.
    Use on larger datasets (>200 samples) for best results.
    """
    model = YOLO("yolov8n-seg.pt")
    model.tune(
        data=data_yaml,
        epochs=30,
        iterations=n_trials,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
    )


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8-seg foot training")
    parser.add_argument("--data", default="foot_dataset/dataset.yaml")
    parser.add_argument("--model", default="yolov8n-seg",
                        choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--stage", type=int, default=0,
                        help="0=full pipeline, 1=stage1 only, 2=stage2 only")
    parser.add_argument("--weights", default=None, help="For stage 2 or export")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.stage == 0:
        final = full_training_pipeline(args.data, args.model, args.device)
    elif args.stage == 1:
        final = train_stage1(args.data, args.model, device=args.device)
    elif args.stage == 2:
        assert args.weights, "--weights required for stage 2"
        final = train_stage2(args.weights, args.data, device=args.device)
    elif args.export:
        assert args.weights, "--weights required for export"
        export_coreml(args.weights)
        export_onnx(args.weights)
