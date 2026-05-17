"""
backend/app/schemas/v2_schemas.py

Pydantic v2 schemas for the depth-only scan pipeline.
"""

from typing import Optional
from pydantic import BaseModel, Field


class V2UploadResponse(BaseModel):
    job_id: str = Field(..., description="16-char hex job identifier")
    message: str
    size_bytes: int


class V2ProcessResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    message: str


class V2StatusResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""
    error: Optional[str] = None


class V2ResultResponse(BaseModel):
    job_id: str
    foot_length_mm: float = Field(..., description="Foot length in millimeters")
    foot_width_mm: float = Field(..., description="Foot width in millimeters")
    foot_height_mm: float = Field(..., description="Foot height in millimeters")
    eu_size_approx: int = Field(..., description="Approximate EU shoe size")
    mesh_vertices: int
    mesh_triangles: int
    method: str = Field(..., description="depth_only_geometric | depth_only_hybrid")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    total_time: float = Field(..., description="Total pipeline time in seconds")
    stl_url: str = Field(..., description="Relative URL to download generated STL")
