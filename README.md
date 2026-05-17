---
title: DigiFoot API
emoji: 🦶
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Depth-only foot scanning API with YOLOv8-seg
---
# ðŸ¦¶ DigiFoot Backend â€” v2.0

> **Production-ready FastAPI backend** combining the legacy orthopedic insole pipeline with the new **depth-only foot scanning v2 pipeline** powered by iPhone TrueDepth / LiDAR + YOLOv8-seg.

---

## ðŸ†• What's New in v2

- **Depth-only foot scanning**: no RGB images required
- **Hybrid segmentation**: geometric (RANSAC + morphology) + YOLOv8-seg refinement
- **Multi-frame fusion**: combines multiple depth frames into one watertight mesh
- **Robust on tiny datasets**: works without YOLO weights (geometric fallback)
- **Real-time ready**: <50ms per frame on iPhone with CoreML export
- **Backwards compatible**: legacy `/upload-scan` etc. still work

---

## ðŸ“ Project Structure

```
backend/
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py                       # FastAPI entry (legacy + v2)
â”‚   â”œâ”€â”€ config.py                     # Central settings
â”‚   â”œâ”€â”€ routes/
â”‚   â”‚   â”œâ”€â”€ upload.py                 # legacy
â”‚   â”‚   â”œâ”€â”€ process.py                # legacy
â”‚   â”‚   â”œâ”€â”€ result.py                 # legacy
â”‚   â”‚   â”œâ”€â”€ download.py               # legacy
â”‚   â”‚   â””â”€â”€ v2_scan.py                # â˜… NEW depth-only endpoints
â”‚   â”œâ”€â”€ schemas/
â”‚   â”‚   â”œâ”€â”€ response_schema.py        # legacy
â”‚   â”‚   â””â”€â”€ v2_schemas.py             # â˜… NEW Pydantic v2 models
â”‚   â”œâ”€â”€ services/
â”‚   â”‚   â”œâ”€â”€ pipeline.py               # legacy mesh pipeline
â”‚   â”‚   â”œâ”€â”€ (existing legacy services)
â”‚   â”‚   â”œâ”€â”€ depth_pipeline.py         # â˜… NEW orchestrator
â”‚   â”‚   â”œâ”€â”€ depth_preprocessing.py    # â˜… NEW depth filter/clean
â”‚   â”‚   â”œâ”€â”€ foot_segmentation.py      # â˜… NEW geometric + YOLO seg
â”‚   â”‚   â””â”€â”€ scan_trigger.py           # â˜… NEW real-time triggering
â”‚   â”œâ”€â”€ recon/
â”‚   â”‚   â”œâ”€â”€ pipeline.py               # (existing recon code)
â”‚   â”‚   â”œâ”€â”€ measurements.py           # (existing)
â”‚   â”‚   â”œâ”€â”€ ml_refine.py              # (existing)
â”‚   â”‚   â”œâ”€â”€ obj_writer.py             # (existing)
â”‚   â”‚   â”œâ”€â”€ uv_bake.py                # (existing)
â”‚   â”‚   â””â”€â”€ reconstruction_3d.py      # â˜… NEW Poisson fusion + measure
â”‚   â”œâ”€â”€ ml/
â”‚   â”‚   â”œâ”€â”€ pointnet_model.py         # (existing)
â”‚   â”‚   â”œâ”€â”€ pointnet2_model.py        # (existing)
â”‚   â”‚   â”œâ”€â”€ arch_classifier.py        # (existing)
â”‚   â”‚   â”œâ”€â”€ pressure_model.py         # (existing)
â”‚   â”‚   â”œâ”€â”€ model_loader.py           # (existing)
â”‚   â”‚   â””â”€â”€ yolo_seg_model.py         # â˜… NEW YOLO singleton
â”‚   â””â”€â”€ utils/
â”‚       â””â”€â”€ (existing helpers)
â”‚
â”œâ”€â”€ ml_training/
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ dataset.py                # (existing)
â”‚   â”‚   â”œâ”€â”€ synthetic_gen.py          # (existing)
â”‚   â”‚   â””â”€â”€ dataset_preparation.py    # â˜… NEW depth dataset builder
â”‚   â”œâ”€â”€ train.py                      # (existing)
â”‚   â”œâ”€â”€ train_yolov8.py               # â˜… NEW YOLOv8-seg 2-stage trainer
â”‚   â””â”€â”€ eval.py                       # (existing)
â”‚
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ migrate_v2.sh                 # â˜… migrate legacy â†’ v2
â”‚   â”œâ”€â”€ setup_deps.sh                 # one-shot setup
â”‚   â”œâ”€â”€ train_all.sh                  # full training pipeline
â”‚   â””â”€â”€ export_coreml.py              # â˜… CoreML export for iOS
â”‚
â”œâ”€â”€ weights/                          # ML model weights (.pt, .pth)
â”œâ”€â”€ scans/                            # uploaded scan dirs (per job_id)
â”œâ”€â”€ stls/                             # generated STL output
â”œâ”€â”€ outputs/                          # intermediate artifacts
â”œâ”€â”€ validation_set/                   # holdout validation data
â”‚
â”œâ”€â”€ requirements.txt                  # merged dependencies
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ .dockerignore
â”œâ”€â”€ test_e2e.py                       # â˜… v2 endpoint integration test
â””â”€â”€ README.md
```

â˜… = added by v2 migration

---

## ðŸš€ Quick Start

### 1. Setup

```bash
cd backend
bash scripts/setup_deps.sh    # creates venv + installs deps
```

Or manually:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: http://localhost:8000/docs
- ReDoc:      http://localhost:8000/redoc

### 3. Test v2 pipeline (synthetic data â€” no real scans needed)

```bash
python test_e2e.py
```

---

## ðŸ³ Docker

```bash
docker build -t digifoot-backend .

docker run -d \
  -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/scans:/app/scans \
  -v $(pwd)/stls:/app/stls \
  --name digifoot \
  digifoot-backend
```

---

## ðŸ”Œ API Endpoints

### Legacy (mesh-based orthopedic pipeline)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload-scan` | Upload mesh ZIP |
| POST | `/process-scan?job_id=` | Start insole pipeline |
| GET  | `/status/{job_id}` | Poll job |
| GET  | `/result/{job_id}` | Get insole results |
| GET  | `/download-stl/{job_id}` | Download insole STL |

### V2 (depth-only foot scanning)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v2/upload-depth-scan` | Upload ZIP of depth frames |
| POST | `/v2/process-depth-scan?job_id=` | Start depth pipeline |
| GET  | `/v2/status/{job_id}` | Poll job |
| GET  | `/v2/result/{job_id}` | Get foot measurements |
| GET  | `/v2/download-stl/{job_id}` | Download foot STL |
| GET  | `/v2/health` | v2 health + YOLO load status |

### V2 Upload Format

The v2 ZIP must contain depth frames:

```
depth_scan.zip
â”œâ”€â”€ depth_0001.png         # 16-bit PNG (depth in mm)
â”œâ”€â”€ depth_0002.png
â”œâ”€â”€ depth_0003.png
â”œâ”€â”€ ...
â””â”€â”€ camera_intrinsics.json  # optional
```

Or alternatively `.npy` files (float32, meters).

`camera_intrinsics.json`:
```json
{"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0}
```

### V2 Response

```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "foot_length_mm": 263.4,
  "foot_width_mm": 97.2,
  "foot_height_mm": 68.1,
  "eu_size_approx": 39,
  "mesh_vertices": 8421,
  "mesh_triangles": 16730,
  "method": "depth_only_hybrid",
  "confidence_score": 0.87,
  "total_time": 4.21,
  "stl_url": "/v2/download-stl/a1b2c3d4e5f6g7h8"
}
```

---

## ðŸ§  Training the YOLOv8-Seg Model

### Full training pipeline

```bash
bash scripts/train_all.sh
```

This runs:
1. **Synthetic data generation** â€” 500 synthetic foot depth maps for pre-training
2. **Stage 1 training** â€” frozen backbone (transfer learning, 150 epochs)
3. **Stage 2 training** â€” full fine-tuning (low LR, 100 epochs)
4. **Export** â€” to CoreML + ONNX
5. **Install** â€” copies weights to `weights/foot_yolov8_seg.pt`

### Manual training

```bash
cd ml_training/

# 1. Generate synthetic dataset
python data/dataset_preparation.py synthetic --output data/foot_dataset --n 500

# 2. Process real depth captures
# (See "Dataset Preparation" in IMPLEMENTATION_GUIDE.md)

# 3. Train
python train_yolov8.py --data data/foot_dataset/dataset.yaml --device 0

# 4. Export
python train_yolov8.py --export --weights foot_scan_runs/foot_seg_stage2/weights/best.pt
```

The pipeline works **without YOLO weights** (geometric mode), so you can deploy immediately and train in parallel.

---

## âš™ï¸ Configuration

All settings in `app/config.py` are env-overridable:

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_DIR` | `weights` | YOLO + ML weights directory |
| `SCANS_DIR` | `scans` | Upload storage |
| `STLS_DIR` | `stls` | STL output |
| `YOLO_MODEL_NAME` | `foot_yolov8_seg.pt` | YOLO weights filename |
| `CAMERA_FX/FY/CX/CY` | `585/585/256/192` | TrueDepth intrinsics |
| `DEPTH_MIN_M` | `0.20` | Min scan distance (m) |
| `DEPTH_MAX_M` | `1.50` | Max scan distance (m) |
| `FLOOR_RANSAC_THRESHOLD` | `0.02` | Floor plane tolerance (m) |
| `RECON_TARGET_TRIANGLES` | `50000` | Mesh decimation target |

---

## ðŸ“± iOS Integration

### Required client flow

```swift
// 1. Capture multiple depth frames during user "FaceID-style" scan
let depthFrames: [Data] = captureDepthFrames()

// 2. ZIP frames + intrinsics
let zipData = makeZip(frames: depthFrames, intrinsics: cameraIntrinsics)

// 3. Upload
let uploadResp = try await api.upload("/v2/upload-depth-scan", zip: zipData)

// 4. Process
try await api.post("/v2/process-depth-scan?job_id=\(uploadResp.jobId)")

// 5. Poll status every 2s
while true {
    let s = try await api.get("/v2/status/\(jobId)")
    if s.status == "completed" { break }
    try await Task.sleep(nanoseconds: 2_000_000_000)
}

// 6. Get measurements + STL
let result = try await api.get("/v2/result/\(jobId)")
let stl = try await api.download("/v2/download-stl/\(jobId)")
```

### Real-time scan triggering (on-device)

Use `app/services/scan_trigger.py` logic ported to Swift for FaceID-style auto-capture (see `IMPLEMENTATION_GUIDE.md` for Swift template).

---

## ðŸ“Š Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Pipeline time | < 8s per scan | 5 frames, single CPU core |
| Inference (CoreML, iPhone 15) | ~20ms/frame | YOLOv8n-seg FP16 + ANE |
| Dimensional accuracy | Â± 5mm | Single frame |
| Dimensional accuracy | Â± 2mm | Multi-frame fusion (10+ frames) |
| Mesh quality | Watertight | Poisson reconstruction |

---

## ðŸ” How It Works

```
TrueDepth/LiDAR frames
       â”‚
       â–¼
Preprocessing  (fill holes, bilateral filter, normalize)
       â”‚
       â–¼
Floor removal  (RANSAC plane detection)
       â”‚
       â–¼
Hybrid segmentation
  â”œâ”€ Geometric (depth threshold + morphology)
  â””â”€ YOLOv8-seg (if model available)
  â†’ AND-combine for highest precision
       â”‚
       â–¼
Multi-frame fusion (if multiple valid frames)
       â”‚
       â–¼
Point cloud cleanup (outlier removal, normals)
       â”‚
       â–¼
Poisson surface reconstruction â†’ watertight mesh
       â”‚
       â–¼
Mesh smoothing + decimation
       â”‚
       â–¼
Measurements + STL export
```

---

## ðŸ› ï¸ Migration from v1

Already running the legacy pipeline? Run the migration script:

```bash
bash scripts/migrate_v2.sh
```

This:
1. Backs up `app/main.py` â†’ `app/main.py.bak`
2. Creates required directories
3. Installs new dependencies
4. Verifies the v2 pipeline loads

Existing legacy endpoints remain unchanged.

---

## ðŸ“„ License

Proprietary. All rights reserved.

---
title: DigiFoot API
emoji: ðŸ¦¶
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Depth-only foot scanning API with YOLOv8-seg
---
