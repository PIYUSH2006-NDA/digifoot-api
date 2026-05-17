# 🦶 DigiFoot Backend — v2.0

> **Production-ready FastAPI backend** combining the legacy orthopedic insole pipeline with the new **depth-only foot scanning v2 pipeline** powered by iPhone TrueDepth / LiDAR + YOLOv8-seg.

---

## 🆕 What's New in v2

- **Depth-only foot scanning**: no RGB images required
- **Hybrid segmentation**: geometric (RANSAC + morphology) + YOLOv8-seg refinement
- **Multi-frame fusion**: combines multiple depth frames into one watertight mesh
- **Robust on tiny datasets**: works without YOLO weights (geometric fallback)
- **Real-time ready**: <50ms per frame on iPhone with CoreML export
- **Backwards compatible**: legacy `/upload-scan` etc. still work

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                       # FastAPI entry (legacy + v2)
│   ├── config.py                     # Central settings
│   ├── routes/
│   │   ├── upload.py                 # legacy
│   │   ├── process.py                # legacy
│   │   ├── result.py                 # legacy
│   │   ├── download.py               # legacy
│   │   └── v2_scan.py                # ★ NEW depth-only endpoints
│   ├── schemas/
│   │   ├── response_schema.py        # legacy
│   │   └── v2_schemas.py             # ★ NEW Pydantic v2 models
│   ├── services/
│   │   ├── pipeline.py               # legacy mesh pipeline
│   │   ├── (existing legacy services)
│   │   ├── depth_pipeline.py         # ★ NEW orchestrator
│   │   ├── depth_preprocessing.py    # ★ NEW depth filter/clean
│   │   ├── foot_segmentation.py      # ★ NEW geometric + YOLO seg
│   │   └── scan_trigger.py           # ★ NEW real-time triggering
│   ├── recon/
│   │   ├── pipeline.py               # (existing recon code)
│   │   ├── measurements.py           # (existing)
│   │   ├── ml_refine.py              # (existing)
│   │   ├── obj_writer.py             # (existing)
│   │   ├── uv_bake.py                # (existing)
│   │   └── reconstruction_3d.py      # ★ NEW Poisson fusion + measure
│   ├── ml/
│   │   ├── pointnet_model.py         # (existing)
│   │   ├── pointnet2_model.py        # (existing)
│   │   ├── arch_classifier.py        # (existing)
│   │   ├── pressure_model.py         # (existing)
│   │   ├── model_loader.py           # (existing)
│   │   └── yolo_seg_model.py         # ★ NEW YOLO singleton
│   └── utils/
│       └── (existing helpers)
│
├── ml_training/
│   ├── data/
│   │   ├── dataset.py                # (existing)
│   │   ├── synthetic_gen.py          # (existing)
│   │   └── dataset_preparation.py    # ★ NEW depth dataset builder
│   ├── train.py                      # (existing)
│   ├── train_yolov8.py               # ★ NEW YOLOv8-seg 2-stage trainer
│   └── eval.py                       # (existing)
│
├── scripts/
│   ├── migrate_v2.sh                 # ★ migrate legacy → v2
│   ├── setup_deps.sh                 # one-shot setup
│   ├── train_all.sh                  # full training pipeline
│   └── export_coreml.py              # ★ CoreML export for iOS
│
├── weights/                          # ML model weights (.pt, .pth)
├── scans/                            # uploaded scan dirs (per job_id)
├── stls/                             # generated STL output
├── outputs/                          # intermediate artifacts
├── validation_set/                   # holdout validation data
│
├── requirements.txt                  # merged dependencies
├── Dockerfile
├── .dockerignore
├── test_e2e.py                       # ★ v2 endpoint integration test
└── README.md
```

★ = added by v2 migration

---

## 🚀 Quick Start

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

### 3. Test v2 pipeline (synthetic data — no real scans needed)

```bash
python test_e2e.py
```

---

## 🐳 Docker

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

## 🔌 API Endpoints

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
├── depth_0001.png         # 16-bit PNG (depth in mm)
├── depth_0002.png
├── depth_0003.png
├── ...
└── camera_intrinsics.json  # optional
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

## 🧠 Training the YOLOv8-Seg Model

### Full training pipeline

```bash
bash scripts/train_all.sh
```

This runs:
1. **Synthetic data generation** — 500 synthetic foot depth maps for pre-training
2. **Stage 1 training** — frozen backbone (transfer learning, 150 epochs)
3. **Stage 2 training** — full fine-tuning (low LR, 100 epochs)
4. **Export** — to CoreML + ONNX
5. **Install** — copies weights to `weights/foot_yolov8_seg.pt`

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

## ⚙️ Configuration

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

## 📱 iOS Integration

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

## 📊 Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Pipeline time | < 8s per scan | 5 frames, single CPU core |
| Inference (CoreML, iPhone 15) | ~20ms/frame | YOLOv8n-seg FP16 + ANE |
| Dimensional accuracy | ± 5mm | Single frame |
| Dimensional accuracy | ± 2mm | Multi-frame fusion (10+ frames) |
| Mesh quality | Watertight | Poisson reconstruction |

---

## 🔍 How It Works

```
TrueDepth/LiDAR frames
       │
       ▼
Preprocessing  (fill holes, bilateral filter, normalize)
       │
       ▼
Floor removal  (RANSAC plane detection)
       │
       ▼
Hybrid segmentation
  ├─ Geometric (depth threshold + morphology)
  └─ YOLOv8-seg (if model available)
  → AND-combine for highest precision
       │
       ▼
Multi-frame fusion (if multiple valid frames)
       │
       ▼
Point cloud cleanup (outlier removal, normals)
       │
       ▼
Poisson surface reconstruction → watertight mesh
       │
       ▼
Mesh smoothing + decimation
       │
       ▼
Measurements + STL export
```

---

## 🛠️ Migration from v1

Already running the legacy pipeline? Run the migration script:

```bash
bash scripts/migrate_v2.sh
```

This:
1. Backs up `app/main.py` → `app/main.py.bak`
2. Creates required directories
3. Installs new dependencies
4. Verifies the v2 pipeline loads

Existing legacy endpoints remain unchanged.

---

## 📄 License

Proprietary. All rights reserved.

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