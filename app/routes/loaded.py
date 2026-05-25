# app/routes/loaded.py
#
# FastAPI routes for the loaded-state ("FaceID for feet") foot scan.
#
#   POST /v2/loaded/reconstruct   - upload the iOS zip, run reconstruction
#   GET  /v2/loaded/mesh/{side}/{scan_id}
#                                 - download the reconstructed .ply mesh
#
# Register in your app entrypoint (e.g. app/main.py):
#     from app.routes import loaded
#     app.include_router(loaded.router)

from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.recon.loaded_reconstruction import reconstruct_loaded_scan
from app.services.loaded_storage import storage
from app.ml.loaded_yolo import get_loaded_yolo
from app.schemas.loaded import LoadedReconstructResponse

router = APIRouter(prefix="/v2/loaded", tags=["loaded"])


@router.post("/reconstruct", response_model=LoadedReconstructResponse)
async def reconstruct(
    scan: UploadFile = File(...),
    side: str = Form("right"),
    scan_id: str | None = Form(None),
):
    """Reconstruct a loaded-state foot scan from the iOS capture zip.

    The Flutter client (LoadedScanService) posts:
        scan     - the LoadedScanEngine zip
        side     - "left" | "right"
        scan_id  - optional; if given and a matching unloaded mesh exists,
                   the loaded reconstruction is aligned to it.

    Reconstruction is CPU-heavy (Open3D registration + Poisson). It runs in
    a worker thread so the event loop is not blocked. On large scans this
    can still exceed a platform request timeout — if that happens, move to
    a background job + polling endpoint.
    """
    saved_id, zip_path = storage.save_upload(scan.file, side)
    out_path = storage.output_mesh_path(side, saved_id)
    unloaded_path = storage.find_unloaded_mesh(scan_id, side)

    try:
        report = await run_in_threadpool(
            reconstruct_loaded_scan,
            zip_path=zip_path,
            out_mesh_path=out_path,
            unloaded_mesh_path=unloaded_path,
            yolo_model=get_loaded_yolo(),
        )
        return LoadedReconstructResponse(
            ok=True,
            side=side,
            angle_count=report.get("angle_count", 0),
            vertices=report.get("vertices", 0),
            triangles=report.get("triangles", 0),
            aligned_to_unloaded=report.get("aligned_to_unloaded", False),
            mesh_url=f"/v2/loaded/mesh/{side}/{saved_id}",
        )
    except Exception as exc:  # noqa: BLE001
        return LoadedReconstructResponse(
            ok=False, side=side, error=str(exc)
        )


@router.get("/mesh/{side}/{scan_id}")
async def get_mesh(side: str, scan_id: str):
    """Download a reconstructed loaded-scan mesh."""
    path = Path(storage.output_mesh_path(side, scan_id))
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "mesh not found"})
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
    )