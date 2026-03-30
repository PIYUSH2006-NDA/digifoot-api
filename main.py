from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
import os
import shutil
import uuid
import trimesh

# ❌ REMOVE THIS FOR NOW
# from backend.sole.pipeline import run_pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/process_multi")
async def process_multi(
    request: Request,
    right_loaded: UploadFile = File(...),
    left_loaded: UploadFile = File(...),
    right_unloaded: UploadFile = File(...),
    left_unloaded: UploadFile = File(...),
    right_insole: UploadFile = File(...),
    left_insole: UploadFile = File(...)
):
    try:
        print("🔥 SAFE MODE RUN (PIPELINE DISABLED)")

        image_paths = []

        def save_file(file):
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            path = os.path.join(UPLOAD_DIR, unique_name)

            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            return path

        # Save all files
        image_paths.append(save_file(right_loaded))
        image_paths.append(save_file(left_loaded))
        image_paths.append(save_file(right_unloaded))
        image_paths.append(save_file(left_unloaded))
        image_paths.append(save_file(right_insole))
        image_paths.append(save_file(left_insole))

        # 🔥 TEMP DUMMY MESH
        mesh = trimesh.creation.box([0.2, 0.1, 0.02])

        glb_name = f"insole_{uuid.uuid4()}.glb"
        glb_path = os.path.join(OUTPUT_DIR, glb_name)

        mesh.export(glb_path)

        base_url = str(request.base_url).rstrip("/")
        file_url = f"{base_url}/download/{glb_name}"

        # Cleanup
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass

        return {
            "status": "success",
            "file_url": file_url,
            "note": "Pipeline temporarily disabled"
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}


@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename
    )
