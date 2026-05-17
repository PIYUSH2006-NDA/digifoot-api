"""
Process route.
POST /process-scan – trigger asynchronous pipeline execution.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.schemas.response_schema import ProcessResponse, ErrorResponse, JobStatus
from app.services.pipeline import get_job, set_job, run_pipeline
from app.utils.storage import get_job_dir, find_mesh_file
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Processing"])


@router.post(
    "/process-scan",
    response_model=ProcessResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Start processing a previously uploaded scan",
)
async def process_scan(
    job_id: str = Query(..., description="Job ID returned by /upload-scan"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if record.status == "processing":
        raise HTTPException(status_code=409, detail="Job is already being processed")

    if record.status == "completed":
        raise HTTPException(status_code=409, detail="Job has already completed")

    # Locate mesh
    job_dir = get_job_dir(job_id)
    extract_dir = job_dir / "extracted"
    mesh_file = find_mesh_file(extract_dir)
    if mesh_file is None:
        raise HTTPException(status_code=404, detail="Mesh file not found for this job")

    log.info("Enqueuing pipeline for job %s (%s)", job_id, mesh_file)

    # Launch in background
    background_tasks.add_task(run_pipeline, job_id, str(mesh_file))

    record.status = "processing"
    record.message = "Processing started"
    set_job(record)

    return ProcessResponse(job_id=job_id, status=JobStatus.processing)
