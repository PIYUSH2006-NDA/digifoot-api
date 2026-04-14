"""
Main processing pipeline.
Orchestrates every stage from raw mesh to final insole STL.
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.services.mesh_cleaner import load_mesh, clean_mesh, mesh_to_points, remove_ground_plane, downsample_points
from app.services.calibration import auto_calibrate, validate_dimensions
from app.services.foot_segmenter import segment_foot, refine_segmentation
from app.services.reconstruction import estimate_normals, reconstruct_mesh, ensure_watertight, smooth_mesh
from app.services.landmark_detector import detect_landmarks
from app.services.biomechanics import run_biomechanical_analysis
from app.services.pressure_analysis import run_pressure_analysis
from app.services.insole_generator import generate_insole
from app.utils.storage import stl_path_for_job, get_job_dir
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# In-memory job store (swap with Redis / DB for production scale-out)
# ---------------------------------------------------------------------------
@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"          # pending | processing | completed | failed
    message: Optional[str] = None
    foot_length_mm: float = 0.0
    foot_width_mm: float = 0.0
    arch_height_mm: float = 0.0
    arch_type: str = ""
    pressure_score: float = 0.0
    confidence_score: float = 0.0
    stl_url: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


_jobs: dict[str, JobRecord] = {}


def get_job(job_id: str) -> Optional[JobRecord]:
    return _jobs.get(job_id)


def set_job(record: JobRecord) -> None:
    _jobs[record.job_id] = record


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(job_id: str, mesh_path: str) -> None:
    """
    Execute the full insole generation pipeline.
    Designed to be called from a BackgroundTask.
    """
    record = get_job(job_id)
    if record is None:
        record = JobRecord(job_id=job_id)
    record.status = "processing"
    record.message = "Pipeline started"
    set_job(record)

    t0 = time.time()
    try:
        # ── 1. Load & clean mesh ──────────────────────────────────────
        log.info("=== [%s] Stage 1: Load & Clean ===", job_id)
        mesh = load_mesh(mesh_path)
        mesh = clean_mesh(mesh)

        # ── 2. Extract points & calibrate ─────────────────────────────
        log.info("=== [%s] Stage 2: Calibrate ===", job_id)
        points = mesh_to_points(mesh)
        points = auto_calibrate(points)
        if not validate_dimensions(points):
            log.warning("[%s] Dimension validation failed - continuing anyway", job_id)

        # ── 3. Remove ground plane ────────────────────────────────────
        log.info("=== [%s] Stage 3: Ground Removal ===", job_id)
        points = remove_ground_plane(points)

        # ── 4. Segment foot ───────────────────────────────────────────
        log.info("=== [%s] Stage 4: Segmentation ===", job_id)
        points = segment_foot(points)
        points = refine_segmentation(points)
        points = downsample_points(points)

        # ── 5. Reconstruct watertight mesh ────────────────────────────
        log.info("=== [%s] Stage 5: Reconstruction ===", job_id)
        normals = estimate_normals(points)
        recon_mesh = reconstruct_mesh(points, normals)
        recon_mesh = ensure_watertight(recon_mesh)
        recon_mesh = smooth_mesh(recon_mesh)

        # ── 6. Landmark detection ─────────────────────────────────────
        log.info("=== [%s] Stage 6: Landmarks ===", job_id)
        landmarks = detect_landmarks(points)

        # ── 7. Biomechanical analysis ─────────────────────────────────
        log.info("=== [%s] Stage 7: Biomechanics ===", job_id)
        bio = run_biomechanical_analysis(points)

        # ── 8. Pressure analysis ──────────────────────────────────────
        log.info("=== [%s] Stage 8: Pressure ===", job_id)
        pressure = run_pressure_analysis(bio.features)

        # ── 9. Generate insole STL ────────────────────────────────────
        log.info("=== [%s] Stage 9: Insole Generation ===", job_id)
        stl_out = stl_path_for_job(job_id)
        generate_insole(landmarks, bio.arch_type, stl_out)

        # ── 10. Finalise ──────────────────────────────────────────────
        elapsed = time.time() - t0
        record.status = "completed"
        record.message = f"Pipeline completed in {elapsed:.1f}s"
        record.foot_length_mm = round(landmarks.foot_length_mm, 2)
        record.foot_width_mm = round(landmarks.foot_width_mm, 2)
        record.arch_height_mm = round(landmarks.arch_height_mm, 2)
        record.arch_type = bio.arch_type
        record.pressure_score = round(pressure.average_score, 4)
        record.confidence_score = round(bio.confidence, 4)
        record.stl_url = f"/download-stl/{job_id}"
        record.completed_at = time.time()
        set_job(record)

        # Persist result metadata alongside the scan
        _save_result_json(job_id, record)

        log.info("=== [%s] DONE (%.1fs) ===", job_id, elapsed)

    except Exception:
        tb = traceback.format_exc()
        log.error("[%s] Pipeline failed:\n%s", job_id, tb)
        record.status = "failed"
        record.error = tb
        record.message = "Pipeline failed"
        set_job(record)


def _save_result_json(job_id: str, record: JobRecord) -> None:
    """Persist result metadata to disk for durability."""
    out = get_job_dir(job_id) / "result.json"
    data = {
        "job_id": record.job_id,
        "foot_length_mm": record.foot_length_mm,
        "foot_width_mm": record.foot_width_mm,
        "arch_height_mm": record.arch_height_mm,
        "arch_type": record.arch_type,
        "pressure_score": record.pressure_score,
        "confidence_score": record.confidence_score,
        "stl_url": record.stl_url,
    }
    out.write_text(json.dumps(data, indent=2))
    log.info("Result metadata saved to %s", out)
