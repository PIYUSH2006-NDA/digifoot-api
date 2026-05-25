# app/schemas/loaded.py
#
# Request/response models for the loaded-state foot scan route.

from pydantic import BaseModel


class LoadedReconstructResponse(BaseModel):
    """Response for POST /v2/loaded/reconstruct."""
    ok: bool
    side: str
    angle_count: int = 0
    vertices: int = 0
    triangles: int = 0
    aligned_to_unloaded: bool = False
    # Relative URL the client can GET to download the mesh (.ply).
    mesh_url: str | None = None
    # Populated only when ok is False.
    error: str | None = None