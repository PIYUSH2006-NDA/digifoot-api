from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from typing import List
import os
import shutil
import uuid
import trimesh

from backend.sole.pipeline import run_pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"status": "API running"}


# ---------------------------
# 🔥 MAIN PROCESS ENDPOINT
# ---------------------------
@app.post("/process_multi")
async def generate_render_model(request: Request, files: List[UploadFile] = File(...)):

    try:
        print("🔥 REAL PIPELINE RUN")

        if not files:
            return {"error": "No files uploaded"}

        image_paths = []

        # ---------------------------
        # SAVE UPLOADED FILES (SAFE)
        # ---------------------------
        for file in files:
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image_paths.append(file_path)

        # ---------------------------
        # RUN PIPELINE
        # ---------------------------
        result = run_pipeline(image_paths, OUTPUT_DIR)

        blender_stl = result.get("blender_stl")
        parametric_stl = result.get("parametric_stl")

        final_stl = blender_stl if blender_stl else parametric_stl

        if final_stl is None:
            return {"error": "No STL generated"}

        # ---------------------------
        # CONVERT STL → GLB
        # ---------------------------
        mesh = trimesh.load(final_stl)

        glb_name = f"insole_{uuid.uuid4()}.glb"
        glb_path = os.path.join(OUTPUT_DIR, glb_name)

        mesh.export(glb_path)

        print("✅ FINAL GLB GENERATED")

        # ---------------------------
        # DYNAMIC BASE URL (IMPORTANT)
        # ---------------------------
        base_url = str(request.base_url).rstrip("/")

        file_url = f"{base_url}/download/{glb_name}"

        # ---------------------------
        # CLEANUP UPLOAD FILES
        # ---------------------------
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass

        return {
            "status": "success",
            "file_url": file_url,
            "analysis": result.get("analysis", {})
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}


# ---------------------------
# 🔥 DOWNLOAD ENDPOINT
# ---------------------------
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
