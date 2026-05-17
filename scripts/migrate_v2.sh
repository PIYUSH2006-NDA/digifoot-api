#!/usr/bin/env bash
# scripts/migrate_v2.sh
# Migrate existing legacy DigiFoot backend to v2 with depth-only pipeline.

set -e

echo "=== DigiFoot v2 Migration ==="

# 1. Backup existing main.py
if [ -f "app/main.py" ]; then
    echo "[1/4] Backing up legacy main.py..."
    cp app/main.py app/main.py.bak
fi

# 2. Ensure required directories exist
echo "[2/4] Creating directories..."
mkdir -p weights scans stls outputs validation_set ml_training/data

# 3. Install missing dependencies
echo "[3/4] Installing v2 dependencies..."
pip install -r requirements.txt

# 4. Smoke test
echo "[4/4] Running smoke test..."
python -c "
from app.services.depth_pipeline import DepthFootPipeline
p = DepthFootPipeline()
print('✓ DepthFootPipeline initialized')
print(f'  YOLO loaded: {p.inference is not None}')
"

echo ""
echo "✓ Migration complete."
echo ""
echo "Next steps:"
echo "  1. Place YOLOv8-seg weights at: weights/foot_yolov8_seg.pt"
echo "  2. Restart server: uvicorn app.main:app --reload"
echo "  3. Test v2 endpoint: curl http://localhost:8000/v2/health"
