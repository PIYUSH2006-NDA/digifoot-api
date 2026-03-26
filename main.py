from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import trimesh
import os

app = FastAPI()

# ✅ CORRECT BASE URL
BASE_URL = "https://something.onrender.com"


@app.post("/process_multi")
async def generate_render_model(files: List[UploadFile] = File(...)):

    try:
        print("🔥 CLEAN RUN")

        mesh = trimesh.creation.box([0.2, 0.1, 0.02])

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        file_name = "insole.glb"
        glb_path = os.path.join(output_dir, file_name)

        mesh.export(glb_path)

        print("✅ GLB GENERATED")

        # ✅ CORRECT URL
        file_url = f"{BASE_URL}/download/{file_name}"

        return {
            "status": "success",
            "file_url": file_url
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}


@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join("outputs", filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

@app.get("/")
def home():
    return {"status": "API running"}

    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename
    )