#!/usr/bin/env bash
# scripts/train_all.sh
# Full training pipeline: synthetic generation → stage 1 → stage 2 → export

set -e

DATASET_DIR=${DATASET_DIR:-ml_training/data/foot_dataset}
N_SYNTHETIC=${N_SYNTHETIC:-500}
MODEL=${MODEL:-yolov8n-seg}
DEVICE=${DEVICE:-0}     # GPU index, or "cpu", or "mps"

echo "=== Full Training Pipeline ==="
echo "Dataset:    $DATASET_DIR"
echo "Synthetic:  $N_SYNTHETIC samples"
echo "Model:      $MODEL"
echo "Device:     $DEVICE"
echo ""

# 1. Generate synthetic dataset (pre-training)
echo "[1/4] Generating synthetic dataset..."
cd ml_training/
python data/dataset_preparation.py synthetic \
    --output "$DATASET_DIR" \
    --n $N_SYNTHETIC

# 2. Stage 1 training (frozen backbone)
echo "[2/4] Stage 1: frozen-backbone training..."
python train_yolov8.py \
    --data "$DATASET_DIR/dataset.yaml" \
    --model $MODEL \
    --stage 1 \
    --device $DEVICE

# 3. Stage 2 fine-tuning
echo "[3/4] Stage 2: full fine-tuning..."
WEIGHTS="foot_scan_runs/foot_seg_stage1/weights/best.pt"
python train_yolov8.py \
    --data "$DATASET_DIR/dataset.yaml" \
    --stage 2 \
    --weights $WEIGHTS \
    --device $DEVICE

# 4. Export to CoreML + ONNX
echo "[4/4] Exporting model..."
FINAL_WEIGHTS="foot_scan_runs/foot_seg_stage2/weights/best.pt"
python train_yolov8.py --export --weights $FINAL_WEIGHTS

# Copy to weights dir
cd ..
cp $FINAL_WEIGHTS weights/foot_yolov8_seg.pt
echo ""
echo "✓ Training complete. Final weights: weights/foot_yolov8_seg.pt"
