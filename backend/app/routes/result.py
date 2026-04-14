"""
Result and status routes.
GET /result/{job_id}  – full analysis results.
GET /status/{job_id}  – lightweight status check.
"""

from fastapi import APIRouter, HTTPException, Request
from app.schemas.response_schema import (
    ResultResponse,
    StatusResponse,
    ErrorResponse,
    JobStatus,
    ArchType,
)
from app.services.pipeline import get_job
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Results"])


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Check processing status",
)
async def get_status(job_id: str):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return StatusResponse(
        job_id=record.job_id,
        status=JobStatus(record.status),
        message=record.message,
    )


@router.get(
    "/result/{job_id}",
    response_model=ResultResponse,
    responses={
        404: {"model": ErrorResponse},
        202: {"model": StatusResponse},
    },
    summary="Retrieve full analysis results",
)
async def get_result(job_id: str, request: Request):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if record.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {record.error or 'Unknown error'}",
        )

    if record.status != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {record.status}. Poll /status/{job_id} to check progress.",
        )

    # Preserve existing behavior but ensure absolute URL
    if record.stl_url.startswith("http"):
        absolute_stl_url = record.stl_url
    else:
        base_url = str(request.base_url).rstrip("/")
        absolute_stl_url = f"{base_url}{record.stl_url}"

    return ResultResponse(
        job_id=record.job_id,
        foot_length_mm=record.foot_length_mm,
        foot_width_mm=record.foot_width_mm,
        arch_height_mm=record.arch_height_mm,
        arch_type=ArchType(record.arch_type),
        pressure_score=record.pressure_score,
        confidence_score=record.confidence_score,
        stl_url=absolute_stl_url,
    )