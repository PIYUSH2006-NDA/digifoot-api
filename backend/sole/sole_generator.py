import numpy as np
import trimesh
import os
import cv2
from scipy.ndimage import gaussian_filter


def generate_sole(mask, output_dir):

    h, w = mask.shape

    # --- Step 1: clean mask ---
    mask = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
    mask = (mask > 0.3).astype(np.float32)

    # --- Step 2: distance transform (THIS IS KEY 🔥) ---
    dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)

    # normalize
    dist = dist / (dist.max() + 1e-6)

    # --- Step 3: create height map ---
    height = 4 + 10 * dist   # base + smooth dome

    # --- Step 4: add arch (controlled, not random) ---
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    arch = np.exp(-((xv - 0.3)**2 / 0.1 + (yv)**2 / 0.5))
    height += 4 * arch * mask

    # --- Step 5: heel cup ---
    heel = np.exp(-((xv + 0.8)**2 / 0.05))
    height += 3 * heel * mask

    # --- Step 6: smooth final ---
    height = gaussian_filter(height, sigma=2)

    # --- Step 7: mesh generation (grid-based, stable) ---
    vertices = []
    faces = []

    for i in range(h - 1):
        for j in range(w - 1):

            if mask[i, j] < 0.1:
                continue

            v1 = [i, j, height[i, j]]
            v2 = [i + 1, j, height[i + 1, j]]
            v3 = [i, j + 1, height[i, j + 1]]
            v4 = [i + 1, j + 1, height[i + 1, j + 1]]

            idx = len(vertices)

            vertices.extend([v1, v2, v3, v4])

            faces.append([idx, idx + 1, idx + 2])
            faces.append([idx + 1, idx + 3, idx + 2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # --- Step 8: cleanup ---
    mesh = mesh.process()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    # --- Step 9: export ---
    output_path = os.path.join(output_dir, "foot_sole.stl")
    mesh.export(output_path)

    return output_path