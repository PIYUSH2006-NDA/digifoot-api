#!/usr/bin/env bash
# scripts/setup_deps.sh
# One-shot dependency installation for the depth-only pipeline.

set -e

echo "=== DigiFoot Backend — Setup ==="

# Detect Python
if command -v python3 &> /dev/null; then
    PY=python3
else
    PY=python
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    $PY -m venv venv
fi

# Activate
source venv/bin/activate || . venv/Scripts/activate

# Install requirements
echo "[2/3] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Optional: CoreML (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[3/3] macOS detected — installing coremltools..."
    pip install coremltools>=7.0
else
    echo "[3/3] Skipping coremltools (macOS only)"
fi

echo ""
echo "✓ Setup complete."
echo ""
echo "Run server:    uvicorn app.main:app --reload --port 8000"
echo "Run tests:     python test_e2e.py"
echo "Train YOLO:    python ml_training/train_yolov8.py --data path/to/dataset.yaml"
