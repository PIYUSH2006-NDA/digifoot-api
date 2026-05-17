"""
backend/app/main.py — PATCH
Add the v2 router. Don't touch existing routers.
"""

# existing imports kept...

from fastapi import FastAPI
from .routes import upload, process, result, download   # legacy
from .routes import v2_scan                              # NEW

app = FastAPI(title="DigiFoot API", version="2.0.0")

# legacy routers (untouched)
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(result.router)
app.include_router(download.router)


# v2 router (new)
app.include_router(v2_scan.router)


@app.get("/health")
async def health(): return {"ok": True, "version": "2.0.0"}
