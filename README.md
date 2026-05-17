# 🦶 Orthopedic Insole Generation Pipeline

> **Production-ready backend** for generating custom orthopedic insoles from LiDAR-based foot scans.

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Central configuration
│   ├── routes/
│   │   ├── upload.py              # POST /upload-scan
│   │   ├── process.py             # POST /process-scan
│   │   ├── result.py              # GET  /result/{job_id}  &  GET /status/{job_id}
│   │   └── download.py            # GET  /download-stl/{job_id}
│   ├── services/
│   │   ├── pipeline.py            # Main orchestrator (9 stages)
│   │   ├── mesh_cleaner.py        # Noise removal, degenerate fix, ground plane
│   │   ├── calibration.py         # Metres → mm conversion & validation
│   │   ├── foot_segmenter.py      # DBSCAN foot isolation
│   │   ├── reconstruction.py      # Poisson surface reconstruction
│   │   ├── landmark_detector.py   # Heel, arch, forefoot, toe detection
│   │   ├── biomechanics.py        # PointNet feature extraction + arch classify
│   │   ├── pressure_analysis.py   # PressureNet plantar analysis
│   │   ├── insole_generator.py    # Parametric insole mesh + STL export
│   │   └── geometry_utils.py      # PCA, sampling, normalisation helpers
│   ├── ml/
│   │   ├── pointnet_model.py      # PointNet architecture (T-Net + encoder)
│   │   ├── arch_classifier.py     # 3-class arch classifier head
│   │   ├── pressure_model.py      # 10-region pressure regression head
│   │   └── model_loader.py        # Singleton loader with GPU/CPU fallback
│   ├── utils/
│   │   ├── storage.py             # Filesystem helpers (upload, extract, STL path)
│   │   └── logger.py              # Structured logging
│   └── schemas/
│       └── response_schema.py     # Pydantic response models
├── weights/                       # Pretrained model weights (.pth)
├── scans/                         # Uploaded scan working directories
├── stls/                          # Generated insole STL files
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & Install

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API docs are available at **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/redoc**.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t insole-pipeline .

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/scans:/app/scans \
  -v $(pwd)/stls:/app/stls \
  --name insole-api \
  insole-pipeline
```

---

## 🔌 API Endpoints

### `POST /upload-scan`

Upload a ZIP archive containing the LiDAR scan.

```bash
curl -X POST http://localhost:8000/upload-scan \
  -F "file=@foot_scan.zip"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "message": "Scan uploaded successfully"
}
```

### `POST /process-scan?job_id=<id>`

Start the processing pipeline.

```bash
curl -X POST "http://localhost:8000/process-scan?job_id=a1b2c3d4e5f6g7h8"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "status": "processing",
  "message": "Processing started"
}
```

### `GET /status/{job_id}`

Poll for processing status.

```bash
curl http://localhost:8000/status/a1b2c3d4e5f6g7h8
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "status": "completed",
  "message": "Pipeline completed in 12.3s"
}
```

### `GET /result/{job_id}`

Retrieve full analysis results.

```bash
curl http://localhost:8000/result/a1b2c3d4e5f6g7h8
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6g7h8",
  "foot_length_mm": 265.40,
  "foot_width_mm": 98.20,
  "arch_height_mm": 18.50,
  "arch_type": "normal",
  "pressure_score": 0.4523,
  "confidence_score": 0.8741,
  "stl_url": "/download-stl/a1b2c3d4e5f6g7h8"
}
```

### `GET /download-stl/{job_id}`

Download the generated insole STL file.

```bash
curl -O http://localhost:8000/download-stl/a1b2c3d4e5f6g7h8
```

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

---

## 🧠 ML Models & Weights

The system uses three PyTorch models:

| Model | File | Input | Output |
|-------|------|-------|--------|
| **PointNet** | `pointnet_foot.pth` | (B, 3, 2048) point cloud | (B, 256) shape features |
| **ArchClassifier** | `arch_classifier.pth` | (B, 256) features | 3-class logits [flat, normal, high] |
| **PressureNet** | `pressure_model.pth` | (B, 256) features | (B, 10) regional pressure scores |

### Adding Pretrained Weights

1. Place `.pth` files in the `weights/` directory.
2. Name them: `pointnet_foot.pth`, `arch_classifier.pth`, `pressure_model.pth`.
3. The system falls back to **random initialisation** if weights are missing.

### Training Your Own Models

- **PointNet**: Train on foot scan point clouds using shape reconstruction or classification loss.
- **ArchClassifier**: Fine-tune on labelled foot scans with flat/normal/high arch labels.
- **PressureNet**: Train with ground-truth pressure mat data mapped to foot regions.

All models support GPU when available and fall back to CPU automatically.

---

## ⚙️ Processing Pipeline

The pipeline runs 9 stages sequentially:

```
1. Load & Clean Mesh     → Remove noise, duplicates, degenerate triangles
2. Scale Calibration     → Auto-detect units, convert to mm
3. Ground Removal        → RANSAC plane detection + removal
4. Foot Segmentation     → DBSCAN clustering, largest cluster = foot
5. 3D Reconstruction     → Poisson surface reconstruction → watertight mesh
6. Landmark Detection    → Heel, arch, forefoot, toe tip identification
7. Biomechanical ML      → PointNet features → arch type classification
8. Pressure Analysis     → PressureNet → 10-region plantar pressure map
9. Insole Generation     → Parametric insole with arch support → STL export
```

---

## 🔧 Configuration

All parameters can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALE_FACTOR` | `1000.0` | Mesh unit conversion factor |
| `VOXEL_DOWNSAMPLE_SIZE` | `0.5` | Voxel size (mm) for downsampling |
| `DBSCAN_EPS` | `5.0` | DBSCAN clustering radius (mm) |
| `DBSCAN_MIN_POINTS` | `100` | Minimum cluster size |
| `POISSON_DEPTH` | `9` | Poisson reconstruction depth |
| `ML_NUM_POINTS` | `2048` | Points sampled for ML input |
| `INSOLE_THICKNESS` | `3.0` | Base insole thickness (mm) |
| `HEEL_CUP_DEPTH` | `12.0` | Heel cup depth (mm) |
| `ARCH_HEIGHT_FLAT` | `8.0` | Arch support for flat feet (mm) |
| `ARCH_HEIGHT_NORMAL` | `15.0` | Arch support for normal arch (mm) |
| `ARCH_HEIGHT_HIGH` | `22.0` | Arch support for high arch (mm) |

---

## 📱 iOS Integration

The iOS app should:

1. **Upload**: ZIP the scan data (`mesh.obj`, `camera_poses.json`, images) and `POST` to `/upload-scan`.
2. **Process**: Call `POST /process-scan?job_id=<id>` to start processing.
3. **Poll**: Poll `GET /status/{job_id}` every 2-3 seconds until `status == "completed"`.
4. **Display**: Fetch results from `GET /result/{job_id}`.
5. **Download**: The STL is available at `GET /download-stl/{job_id}`.

All responses are JSON. The STL download is a binary file stream.

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Processing time | < 30 seconds per scan |
| Dimensional accuracy | ≤ 1 mm error |
| STL quality | Watertight, manifold, 3D-print ready |

---

## 📄 License

This project is proprietary. All rights reserved.
