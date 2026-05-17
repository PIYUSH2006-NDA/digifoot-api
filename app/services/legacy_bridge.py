"""
backend/app/services/legacy_bridge.py
Bridge v6 mesh+measurements into existing insole-STL pipeline.
"""
from pathlib import Path


def build_insole_from_mesh(mesh_path: str, side: str, out_path: str,
                           measurements: dict):
    """
    Replace this body with a call into your existing insole builder.
    Example:
        from ..ml.insole_builder import build_stl
        build_stl(mesh_path=mesh_path, side=side, out=out_path,
                  length_mm=measurements["length_mm"],
                  ball_width_mm=measurements["ball_width_mm"],
                  arch_height_mm=measurements["arch_height_mm"])
    """
    raise NotImplementedError(
        "wire to existing insole STL builder in app/ml/insole_builder.py"
    )
