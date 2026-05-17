"""
backend/app/main.py

FastAPI entry point. Wires both legacy and v2 routers.

Note: Legacy routers (upload/process/result/download) call your existing
pipeline.py for the mesh-based orthopedic flow. The v2_scan router uses
the new depth-only pipeline (services/depth_pipeline.py).
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import v2_scan

# Try importing legacy routes (graceful skip if not present yet)
# Legacy mesh routes disabled — requires ML weights not present in HF deploy
LEGACY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
#  APP
# ======================================================================

app = FastAPI(
    title="DigiFoot API",
    description="Depth-only foot scanning + orthopedic insole pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Legacy routers (orthopedic insole) ───────────────────────────────
if LEGACY_AVAILABLE:
    app.include_router(upload.router)
    app.include_router(process.router)
    app.include_router(result.router)
    app.include_router(download.router)
    logger.info("✓ Legacy routers loaded (insole pipeline)")
else:
    logger.warning("⚠ Legacy routers not found — v2 only mode")

# ── V2 router (depth-only scanning) ──────────────────────────────────
app.include_router(v2_scan.router)
logger.info("✓ V2 router loaded (depth-only foot scanning)")


# ======================================================================
#  ROOT / HEALTH
# ======================================================================

@app.get("/")
async def root():
    return {
        "name": "DigiFoot API",
        "version": "2.0.0",
        "endpoints": {
            "legacy": "/upload-scan, /process-scan, /result/{job_id}, /download-stl/{job_id}",
            "v2": "/v2/upload-depth-scan, /v2/process-depth-scan, /v2/result/{job_id}",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    return {"ok": True, "version": "2.0.0", "legacy": LEGACY_AVAILABLE}
