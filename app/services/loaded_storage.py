# app/services/loaded_storage.py
#
# Filesystem storage for the loaded-state scan pipeline:
#   - save the uploaded zip
#   - hand out output mesh paths
#   - locate the matching unloaded mesh (for loaded<->unloaded alignment)
#
# NOTE for Hugging Face Spaces: the container filesystem is ephemeral and
# resets on restart. For durable storage, point BASE_DIR at a mounted
# persistent volume, or push results to external object storage.

import os
import shutil
import uuid
from pathlib import Path

BASE_DIR = Path(os.environ.get("LOADED_DATA_DIR", "data/loaded"))
# Where the unloaded pipeline writes its meshes — adjust to your layout.
UNLOADED_MESH_DIR = Path(os.environ.get("UNLOADED_MESH_DIR", "data/unloaded/outputs"))


class LoadedStorage:
    def __init__(self) -> None:
        self.uploads = BASE_DIR / "uploads"
        self.outputs = BASE_DIR / "outputs"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_obj, side: str) -> tuple[str, str]:
        """Persist the uploaded zip. Returns (scan_id, zip_path)."""
        scan_id = uuid.uuid4().hex[:12]
        zip_path = self.uploads / f"loaded_{side}_{scan_id}.zip"
        with open(zip_path, "wb") as out:
            shutil.copyfileobj(file_obj, out)
        return scan_id, str(zip_path)

    def output_mesh_path(self, side: str, scan_id: str) -> str:
        """Path the reconstructed mesh should be written to."""
        return str(self.outputs / f"loaded_{side}_{scan_id}.ply")

    def find_unloaded_mesh(self, scan_id: str | None, side: str) -> str | None:
        """Locate this user's unloaded foot mesh so the loaded reconstruction
        can be aligned to it. Returns None if not found — alignment is then
        skipped and the loaded mesh is returned in its own frame.

        Wire the lookup below to however the unloaded pipeline names/stores
        its meshes.
        """
        if not scan_id:
            return None
        candidate = UNLOADED_MESH_DIR / f"unloaded_{side}_{scan_id}.ply"
        return str(candidate) if candidate.exists() else None


# Module-level singleton — import this.
storage = LoadedStorage()