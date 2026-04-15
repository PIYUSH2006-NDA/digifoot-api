"""
Upload route.
POST /upload-scan – accept a zipped LiDAR scan package.

MODIFIED: Accepts `foot_side` query parameter ("left" | "right").
The iOS app calls this endpoint TWICE — once per foot — so each
foot gets its own job_id and its own processing pipeline.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional

from app.schemas.response_schema import UploadResponse, ErrorResponse
from app.utils.storage import generate_job_id, save_upload, find_mesh_file
from app.services.pipeline import JobRecord, set_job
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Upload"])


@router.post(
    "/upload-scan",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Upload a zipped LiDAR foot scan",
    description=(
        "Accepts a ZIP archive containing:\n"
        "- `mesh.obj` or `mesh.ply` – the 3-D foot mesh\n"
        "- `camera_poses.json` (optional)\n"
        "- scan images (optional)\n\n"
        "Pass `foot_side=left` or `foot_side=right` to tag the scan.\n"
        "Call this endpoint TWICE (once per foot) to get separate job IDs.\n\n"
        "Returns a `job_id` used for all subsequent requests."
    ),
)
async def upload_scan(
    file: UploadFile = File(...),
    foot_side: Optional[str] = Query(
        None,
        description="Which foot: 'left' or 'right'. "
                    "If omitted, defaults to 'left' for backward compatibility.",
    ),
):
    # Validate content type
    if file.content_type not in (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a ZIP file, got {file.content_type}",
        )

    # Validate foot_side
    side = (foot_side or "left").lower().strip()
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail=f"foot_side must be 'left' or 'right', got '{foot_side}'",
        )

    job_id = generate_job_id()
    log.info(
        "New upload -> job_id=%s  foot_side=%s  filename=%s",
        job_id, side, file.filename,
    )

    try:
        contents = await file.read()
        extract_dir = save_upload(job_id, contents, file.filename or "scan.zip")
    except Exception as exc:
        log.error("Upload failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to process upload: {exc}")

    # Quick validation: does the archive contain a mesh?
    mesh_file = find_mesh_file(extract_dir)
    if mesh_file is None:
        raise HTTPException(
            status_code=400,
            detail="No .obj or .ply mesh file found inside the uploaded archive",
        )

    # Register job — now includes foot_side
    record = JobRecord(
        job_id=job_id,
        status="pending",
        message="Scan uploaded",
        foot_side=side,
    )
    set_job(record)

    return UploadResponse(job_id=job_id, foot_side=side)
