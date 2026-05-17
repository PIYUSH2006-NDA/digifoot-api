"""
backend/app/services/job_store.py
Simple JSON-backed job store with file lock.
"""
import json
import threading
from pathlib import Path


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _write(self, d: dict):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self.path)

    def put(self, job_id: str, data: dict):
        with self.lock:
            d = self._read()
            d[job_id] = data
            self._write(d)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            return self._read().get(job_id)

    def update(self, job_id: str, **kwargs):
        with self.lock:
            d = self._read()
            if job_id in d:
                d[job_id].update(kwargs)
                self._write(d)
