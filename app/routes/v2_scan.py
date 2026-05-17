"""
backend/app/routes/v2_scan.py

v2 API — Depth-only foot scanning endpoints.
Coexists with legacy mesh-based routes (upload.py, process.py, etc.).

Endpoints:
  POST   /v2/upload-depth-scan      → upload ZIP of depth frames
  POST   /v2/process-depth-scan     → start depth-only pipeline
  GET    /v2/status/{job_id}        → poll job status
  GET    /v2/result/{job_id}        → fetch results
  GET    /v2/download-stl/{job_id}  → download generated foot STL
"""

import os
import uuid
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from ..services.depth_pipeline import get_depth_pipeline
from ..schemas.v2_schemas import (
    V2UploadResponse,
    V2ProcessResponse,
    V2StatusResponse,
    V2ResultResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["v2-depth-scan"])

# Simple in-memory job store (replace with Redis/Celery for production)
_jobs: Dict[str, Dict[str, Any]] = {}

# Storage paths (use your config in production)
SCANS_DIR = Path(os.getenv("SCANS_DIR", "scans"))
STLS_DIR = Path(os.getenv("STLS_DIR", "stls"))
SCANS_DIR.mkdir(parents=True, exist_ok=True)
STLS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
#  UPLOAD
# ======================================================================

@router.post("/upload-depth-scan", response_model=V2UploadResponse)
async def upload_depth_scan(file: UploadFile = File(...)):
    """
    Upload a ZIP containing depth frames.

    Expected ZIP contents:
      depth_0001.png  (16-bit PNG, depth in mm)
      depth_0002.png
      ...
      OR
      *.npy           (float32 depth in meters)
      camera_intrinsics.json (optional — overrides defaults)
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip")

    job_id = uuid.uuid4().hex[:16]
    job_dir = SCANS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / "scan.zip"
    try:
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(job_dir)

        _jobs[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "scan_dir": str(job_dir),
            "message": "Scan uploaded",
        }
        logger.info(f"[{job_id}] Uploaded {len(content):,} bytes")

        return V2UploadResponse(
            job_id=job_id,
            message="Depth scan uploaded successfully",
            size_bytes=len(content),
        )
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
#  PROCESS
# ======================================================================

def _run_depth_pipeline(job_id: str):
    """Background task — runs the full depth pipeline."""
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        job["status"] = "processing"
        scan_dir = job["scan_dir"]
        stl_path = STLS_DIR / f"{job_id}.stl"

        pipeline = get_depth_pipeline()
        result = pipeline.run(
            job_id=job_id,
            scan_dir=scan_dir,
            stl_out_path=str(stl_path),
        )

        job.update(result)
        job["status"] = "completed"
        logger.info(f"[{job_id}] ✓ Completed in {result.get('total_time')}s")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed")
        job["status"] = "failed"
        job["error"] = str(e)


@router.post("/process-depth-scan", response_model=V2ProcessResponse)
async def process_depth_scan(job_id: str, background_tasks: BackgroundTasks):
    """Start depth pipeline as a background task."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job_id not found")
    if _jobs[job_id]["status"] not in ("uploaded", "failed"):
        return V2ProcessResponse(
            job_id=job_id,
            status=_jobs[job_id]["status"],
            message=f"Job already {_jobs[job_id]['status']}",
        )

    background_tasks.add_task(_run_depth_pipeline, job_id)
    _jobs[job_id]["status"] = "queued"

    return V2ProcessResponse(
        job_id=job_id,
        status="queued",
        message="Depth pipeline queued for processing",
    )


# ======================================================================
#  STATUS / RESULT / DOWNLOAD
# ======================================================================

@router.get("/status/{job_id}", response_model=V2StatusResponse)
async def get_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job_id not found")
    job = _jobs[job_id]
    return V2StatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message", ""),
        error=job.get("error"),
    )


@router.get("/result/{job_id}", response_model=V2ResultResponse)
async def get_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job_id not found")
    job = _jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status={job['status']})",
        )

    return V2ResultResponse(
        job_id=job_id,
        foot_length_mm=job.get("foot_length_mm", 0.0),
        foot_width_mm=job.get("foot_width_mm", 0.0),
        foot_height_mm=job.get("foot_height_mm", 0.0),
        eu_size_approx=job.get("eu_size_approx", 0),
        mesh_vertices=job.get("mesh_vertices", 0),
        mesh_triangles=job.get("mesh_triangles", 0),
        method=job.get("method", "depth_only_geometric"),
        confidence_score=job.get("confidence_score", 0.0),
        total_time=job.get("total_time", 0.0),
        stl_url=job.get("stl_url", f"/v2/download-stl/{job_id}"),
    )


@router.get("/download-stl/{job_id}")
async def download_stl(job_id: str):
    stl_path = STLS_DIR / f"{job_id}.stl"
    if not stl_path.exists():
        raise HTTPException(status_code=404, detail="STL not found")
    return FileResponse(
        path=str(stl_path),
        media_type="application/octet-stream",
        filename=f"foot_{job_id}.stl",
    )


# ======================================================================
#  HEALTH (v2-specific)
# ======================================================================

@router.get("/health")
async def v2_health():
    pipeline = get_depth_pipeline()
    return {
        "ok": True,
        "version": "v2-depth-only",
        "yolo_loaded": pipeline.inference is not None,
        "active_jobs": len(_jobs),
    }
