"""
backend/test_e2e.py

End-to-end test for the v2 depth-only scan pipeline.
Generates synthetic depth frames, uploads via REST API, validates result.

Usage:
    # Start server first:
    uvicorn app.main:app --port 8000

    # Then in another terminal:
    python test_e2e.py
"""

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import cv2
import requests

API = os.getenv("API_URL", "http://localhost:8000")
SYNTHETIC_FRAMES = 5  # number of frames to generate


# ======================================================================
#  Helpers
# ======================================================================

def generate_synthetic_foot_depth(h: int = 480, w: int = 640, seed: int = 0):
    """Generate synthetic depth map of a foot (16-bit PNG bytes)."""
    np.random.seed(seed)
    base_depth = 0.60
    floor_depth = 0.65

    depth = np.full((h, w), floor_depth, dtype=np.float32)

    cx = int(w * np.random.uniform(0.40, 0.60))
    cy = int(h * np.random.uniform(0.40, 0.60))
    foot_w = int(np.random.uniform(60, 80))
    foot_h = int(np.random.uniform(150, 200))
    angle = np.random.uniform(-25, 25)

    cv2.ellipse(depth, (cx, cy), (foot_w, foot_h), angle, 0, 360, base_depth, -1)

    # Heel
    heel_x = cx + int(foot_h * 0.55 * np.sin(np.radians(angle)))
    heel_y = cy + int(foot_h * 0.55 * np.cos(np.radians(angle)))
    cv2.ellipse(
        depth, (heel_x, heel_y),
        (int(foot_w * 1.1), int(foot_w * 0.9)),
        angle, 0, 360, base_depth + 0.005, -1,
    )

    # Toes
    front_x = cx - int(foot_h * 0.55 * np.sin(np.radians(angle)))
    front_y = cy - int(foot_h * 0.55 * np.cos(np.radians(angle)))
    perp = np.radians(angle + 90)
    for i in range(5):
        offset = (i - 2) * (foot_w * 2 / 5)
        tx = int(front_x + offset * np.cos(perp))
        ty = int(front_y + offset * np.sin(perp))
        cv2.circle(depth, (tx, ty), 12, base_depth - 0.008, -1)

    # Noise
    depth += np.random.normal(0, 0.003, (h, w)).astype(np.float32)

    # Convert to uint16 (mm)
    depth_uint16 = (depth * 1000).astype(np.uint16)

    # Encode as PNG
    ok, buf = cv2.imencode(".png", depth_uint16)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def build_test_zip() -> bytes:
    """Build a ZIP with multiple synthetic depth frames."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(SYNTHETIC_FRAMES):
            png_bytes = generate_synthetic_foot_depth(seed=i)
            zf.writestr(f"depth_{i:04d}.png", png_bytes)
        # Optional intrinsics
        intrinsics = {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0}
        zf.writestr("camera_intrinsics.json", json.dumps(intrinsics))
    buf.seek(0)
    return buf.read()


# ======================================================================
#  Test
# ======================================================================

def test_v2_pipeline():
    print("=" * 60)
    print("  DigiFoot V2 — Depth Pipeline E2E Test")
    print("=" * 60)

    # 1. Health
    print("\n[1/6] Health check...")
    r = requests.get(f"{API}/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    print(f"  ✓ {r.json()}")

    r = requests.get(f"{API}/v2/health")
    print(f"  ✓ v2: {r.json()}")

    # 2. Build synthetic ZIP
    print(f"\n[2/6] Generating {SYNTHETIC_FRAMES} synthetic depth frames...")
    zip_bytes = build_test_zip()
    print(f"  ✓ ZIP size: {len(zip_bytes):,} bytes")

    # 3. Upload
    print("\n[3/6] Uploading scan...")
    r = requests.post(
        f"{API}/v2/upload-depth-scan",
        files={"file": ("depth_scan.zip", zip_bytes, "application/zip")},
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    job_id = r.json()["job_id"]
    print(f"  ✓ job_id = {job_id}")

    # 4. Process
    print("\n[4/6] Starting processing...")
    r = requests.post(f"{API}/v2/process-depth-scan", params={"job_id": job_id})
    assert r.status_code == 200, f"Process failed: {r.text}"
    print(f"  ✓ {r.json()}")

    # 5. Poll
    print("\n[5/6] Polling status...")
    status = None
    for i in range(120):
        time.sleep(1)
        r = requests.get(f"{API}/v2/status/{job_id}")
        body = r.json()
        status = body["status"]
        if status in ("completed", "failed"):
            print(f"  [{i+1}s] status = {status}")
            if status == "failed":
                print(f"  Error: {body.get('error')}")
            break
        if i % 5 == 0:
            print(f"  [{i+1}s] status = {status}")

    assert status == "completed", f"Pipeline did not complete: {body}"

    # 6. Results
    print("\n[6/6] Fetching results...")
    r = requests.get(f"{API}/v2/result/{job_id}")
    assert r.status_code == 200, f"Result failed: {r.text}"
    result = r.json()
    print(json.dumps(result, indent=2))

    # Validations
    assert result["foot_length_mm"] > 50, "foot_length_mm too small"
    assert result["foot_width_mm"] > 30, "foot_width_mm too small"
    assert result["mesh_vertices"] > 100, "mesh too small"
    assert 0 <= result["confidence_score"] <= 1

    # Download STL
    print("\nDownloading STL...")
    r = requests.get(f"{API}{result['stl_url']}")
    assert r.status_code == 200
    print(f"  ✓ STL size: {len(r.content):,} bytes")

    out_path = Path("test_output.stl")
    out_path.write_bytes(r.content)
    print(f"  ✓ Saved: {out_path}")

    print("\n" + "=" * 60)
    print("  ✓ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_v2_pipeline()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"\n✗ Cannot connect to {API}")
        print("  Start the server first: uvicorn app.main:app --port 8000")
        sys.exit(1)
