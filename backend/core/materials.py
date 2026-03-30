import numpy as np


def apply_material_zones(mesh, analysis):

    vertices = mesh.vertices.copy()

    print("🧱 Applying material intelligence...")

    for i, v in enumerate(vertices):
        x, y, z = v

        # ---------------------------
        # HEEL (SOFT)
        # ---------------------------
        if x < 0.2:
            vertices[i][2] -= 0.01  # softer → compress

        # ---------------------------
        # ARCH (SUPPORT)
        # ---------------------------
        elif 0.3 < x < 0.6:
            vertices[i][2] += 0.015  # firm support

        # ---------------------------
        # FOREFOOT (FLEX)
        # ---------------------------
        elif x > 0.7:
            vertices[i][2] -= 0.005  # flexible region

    mesh.vertices = vertices

    print("✅ Material zones applied")

    return mesh