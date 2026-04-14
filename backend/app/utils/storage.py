"""
Local filesystem storage helpers.
Handles scan uploads, job directories, and STL retrieval.
"""

import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from app.config import SCANS_DIR, STLS_DIR
from app.utils.logger import get_logger

log = get_logger(__name__)


def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return uuid.uuid4().hex[:16]


def get_job_dir(job_id: str) -> Path:
    """Return (and create) the working directory for a given job."""
    job_dir = SCANS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def save_upload(job_id: str, file_bytes: bytes, filename: str) -> Path:
    """
    Persist an uploaded zip archive and extract its contents.
    Returns the extraction directory.
    """
    job_dir = get_job_dir(job_id)
    zip_path = job_dir / filename

    # Write raw bytes
    zip_path.write_bytes(file_bytes)
    log.info("Saved upload %s (%d bytes)", zip_path, len(file_bytes))

    # Extract
    extract_dir = job_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    log.info("Extracted %d entries to %s", len(list(extract_dir.rglob("*"))), extract_dir)
    return extract_dir


def find_mesh_file(extract_dir: Path) -> Optional[Path]:
    """Locate the primary mesh file (.obj or .ply) inside the extracted scan."""
    for ext in ("*.obj", "*.ply"):
        matches = list(extract_dir.rglob(ext))
        if matches:
            log.info("Found mesh file: %s", matches[0])
            return matches[0]
    return None


def stl_path_for_job(job_id: str) -> Path:
    """Return the expected STL output path for a job."""
    return STLS_DIR / f"{job_id}.stl"


def cleanup_job(job_id: str) -> None:
    """Remove all artefacts associated with a job."""
    job_dir = SCANS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    stl = stl_path_for_job(job_id)
    if stl.exists():
        stl.unlink()
    log.info("Cleaned up job %s", job_id)
