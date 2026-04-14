"""
End-to-end integration test.
Generates a synthetic foot mesh, runs the full pipeline via the REST API,
and validates the response.
"""

import io
import json
import time
import zipfile
import requests
import numpy as np
import trimesh

API = "http://localhost:8000"


def make_synthetic_foot() -> bytes:
    """
    Create a simplified foot-shaped mesh (elongated ellipsoid ~260mm × 95mm × 70mm)
    in metres (as a real LiDAR scan would).  Returns an in-memory OBJ zip.
    """
    # Parametric foot shape (elongated ellipsoid)
    length_m = 0.260   # 260 mm
    width_m  = 0.095   # 95 mm
    height_m = 0.070   # 70 mm

    # Generate ellipsoid
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.array(sphere.vertices, dtype=np.float64)
    verts[:, 0] *= length_m / 2  # X = length
    verts[:, 1] *= width_m / 2   # Y = width
    verts[:, 2] *= height_m / 2  # Z = height
    # Shift so bottom sits at Z=0 (simulating foot on ground)
    verts[:, 2] -= verts[:, 2].min()

    mesh = trimesh.Trimesh(vertices=verts, faces=sphere.faces)

    # Add a ground plane (flat square at Z=0) to test ground removal
    ground = trimesh.creation.box(extents=[0.5, 0.5, 0.001])
    ground.apply_translation([0, 0, -0.0005])
    combined = trimesh.util.concatenate([mesh, ground])

    # Export as OBJ
    obj_data = combined.export(file_type="obj")

    # Camera poses JSON (minimal placeholder)
    camera_poses = {
        "frames": [
            {
                "intrinsics": {"fx": 1000, "fy": 1000, "cx": 320, "cy": 240},
                "extrinsics": {"rotation": [[1,0,0],[0,1,0],[0,0,1]], "translation": [0,0,0.5]},
            }
        ]
    }

    # Pack into ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mesh.obj", obj_data if isinstance(obj_data, str) else obj_data.decode())
        zf.writestr("camera_poses.json", json.dumps(camera_poses))
    buf.seek(0)
    return buf.read()


def test_pipeline():
    print("=" * 60)
    print("  ORTHOPEDIC INSOLE PIPELINE – E2E TEST")
    print("=" * 60)

    # 1. Health check
    print("\n[1/6] Health check …")
    r = requests.get(f"{API}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"  ✓ {r.json()}")

    # 2. Generate synthetic scan
    print("\n[2/6] Generating synthetic foot scan …")
    zip_bytes = make_synthetic_foot()
    print(f"  ✓ ZIP size: {len(zip_bytes):,} bytes")

    # 3. Upload
    print("\n[3/6] Uploading scan …")
    r = requests.post(
        f"{API}/upload-scan",
        files={"file": ("foot_scan.zip", zip_bytes, "application/zip")},
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    job_id = r.json()["job_id"]
    print(f"  ✓ job_id = {job_id}")

    # 4. Start processing
    print("\n[4/6] Starting processing …")
    r = requests.post(f"{API}/process-scan", params={"job_id": job_id})
    assert r.status_code == 200, f"Process failed: {r.text}"
    print(f"  ✓ {r.json()}")

    # 5. Poll status
    print("\n[5/6] Polling status …")
    for i in range(60):
        time.sleep(1)
        r = requests.get(f"{API}/status/{job_id}")
        status = r.json()["status"]
        print(f"  [{i+1}s] status = {status}")
        if status in ("completed", "failed"):
            break
    assert status == "completed", f"Pipeline did not complete: {r.json()}"

    # 6. Get results
    print("\n[6/6] Fetching results …")
    r = requests.get(f"{API}/result/{job_id}")
    assert r.status_code == 200, f"Result failed: {r.text}"
    result = r.json()
    print(json.dumps(result, indent=2))

    # Validate result fields
    assert result["foot_length_mm"] > 0, "foot_length_mm should be > 0"
    assert result["foot_width_mm"] > 0, "foot_width_mm should be > 0"
    assert result["arch_type"] in ("flat", "normal", "high"), f"Bad arch_type: {result['arch_type']}"
    assert 0 <= result["pressure_score"] <= 1, "pressure_score out of range"
    assert 0 <= result["confidence_score"] <= 1, "confidence_score out of range"

    # Download STL
    print("\nDownloading STL …")
    r = requests.get(f"{API}{result['stl_url']}")
    assert r.status_code == 200, f"STL download failed: {r.text}"
    print(f"  ✓ STL size: {len(r.content):,} bytes")

    # Validate STL is a valid mesh
    stl_mesh = trimesh.load(io.BytesIO(r.content), file_type="stl")
    print(f"  ✓ STL vertices: {len(stl_mesh.vertices)}, faces: {len(stl_mesh.faces)}")
    print(f"  ✓ Watertight: {stl_mesh.is_watertight}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
