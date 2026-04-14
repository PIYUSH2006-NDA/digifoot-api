"""
FastAPI application entry point.
Assembles all routers and configures middleware.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.routes import upload, process, result, download
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    log.info("[START] %s v%s starting up", API_TITLE, API_VERSION)
    yield
    log.info("[STOP]  %s shutting down", API_TITLE)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS - allow the iOS app (and any other origin during development)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(result.router)
app.include_router(download.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"], summary="Health check")
async def health():
    return {"status": "ok", "version": API_VERSION}
