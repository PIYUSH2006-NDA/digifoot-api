"""
Download route.
GET /download-stl/{job_id} – stream the generated insole STL.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.response_schema import ErrorResponse
from app.services.pipeline import get_job
from app.utils.storage import stl_path_for_job
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Download"])


@router.get(
    "/download-stl/{job_id}",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Download the generated insole STL file",
)
async def download_stl(job_id: str):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if record.status != "completed":
        raise HTTPException(
            status_code=404,
            detail=f"STL not ready – job status is '{record.status}'",
        )

    stl = stl_path_for_job(job_id)
    if not stl.exists():
        raise HTTPException(status_code=404, detail="STL file not found on disk")

    log.info("Serving STL for job %s (%s)", job_id, stl)
    return FileResponse(
        path=str(stl),
        filename=f"insole_{job_id}.stl",
        media_type="application/sla",
    )
