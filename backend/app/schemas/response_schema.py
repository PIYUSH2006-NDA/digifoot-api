"""
Pydantic response models for the REST API.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ArchType(str, Enum):
    flat = "flat"
    normal = "normal"
    high = "high"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    message: str = Field(default="Scan uploaded successfully")


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None


class ProcessResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.processing
    message: str = Field(default="Processing started")


class ResultResponse(BaseModel):
    job_id: str
    foot_length_mm: float = Field(..., description="Foot length in millimetres")
    foot_width_mm: float = Field(..., description="Foot width in millimetres")
    arch_height_mm: float = Field(..., description="Arch height in millimetres")
    arch_type: ArchType = Field(..., description="Classified arch type")
    pressure_score: float = Field(..., description="Average plantar pressure score (0-1)")
    confidence_score: float = Field(..., description="ML prediction confidence (0-1)")
    stl_url: str = Field(..., description="URL to download the generated insole STL")


class ErrorResponse(BaseModel):
    detail: str
