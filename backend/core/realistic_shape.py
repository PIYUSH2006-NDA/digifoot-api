import numpy as np


def apply_realistic_shape(mesh):

    vertices = mesh.vertices.copy()

    print("🦶 Applying realistic sole shaping...")

    for i, v in enumerate(vertices):
        x, y, z = v

        # Normalize coordinates (0 → 1)
        x_norm = normalize(x, vertices[:, 0])
        y_norm = normalize(y, vertices[:, 1])

        # ---------------------------
        # 1. TOE SPRING (front lift)
        # ---------------------------
        if x_norm > 0.75:
            lift = (x_norm - 0.75) * 0.05
            vertices[i][2] += lift

        # ---------------------------
        # 2. HEEL ROUNDING
        # ---------------------------
        if x_norm < 0.2:
            curve = (0.2 - x_norm) * 0.03
            vertices[i][2] += curve

        # ---------------------------
        # 3. EDGE TAPER (remove sharp edges)
        # ---------------------------
        edge_dist = min(y_norm, 1 - y_norm)
        taper = edge_dist * 0.02
        vertices[i][2] -= taper

        # ---------------------------
        # 4. NATURAL FOOT CURVE
        # ---------------------------
        arch_curve = np.sin(np.pi * x_norm) * 0.01
        vertices[i][2] += arch_curve

    mesh.vertices = vertices

    print("✅ Realistic shaping applied")

    return mesh


# ---------------------------
# NORMALIZATION HELPER
# ---------------------------
def normalize(value, arr):
    return (value - arr.min()) / (arr.max() - arr.min() + 1e-6)