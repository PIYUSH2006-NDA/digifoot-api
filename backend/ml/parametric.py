import numpy as np
import cv2
import trimesh


# ---------------------------
# NORMALIZE MASK
# ---------------------------
def preprocess_mask(mask, size=256):
    mask = cv2.resize(mask, (size, size))
    mask = (mask > 0).astype(np.float32)
    return mask


# ---------------------------
# HEIGHT MAP GENERATION
# ---------------------------
def generate_height_map(mask, analysis):
    h, w = mask.shape

    height_map = np.zeros((h, w))

    # Base thickness
    base_thickness = 2.0

    # Arch support
    arch_height = analysis["arch_height"]

    # Create Gaussian arch bump
    center_x = w // 2
    center_y = int(h * 0.5)

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            dx = (x - center_x) / w
            dy = (y - center_y) / h

            arch = np.exp(-(dx**2 + dy**2) * 20)

            height_map[y, x] = base_thickness + arch * arch_height

    return height_map


# ---------------------------
# HEEL CUP
# ---------------------------
def apply_heel_cup(height_map, mask):
    h, w = height_map.shape

    for y in range(int(h * 0.1)):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            depth = (1 - y / (h * 0.1)) * 3.0
            height_map[y, x] += depth

    return height_map


# ---------------------------
# SMOOTHING
# ---------------------------
def smooth_height_map(height_map):
    return cv2.GaussianBlur(height_map, (11, 11), 0)


# ---------------------------
# HEIGHT → MESH
# ---------------------------
def heightmap_to_mesh(height_map, scale=1.0):
    h, w = height_map.shape

    vertices = []
    faces = []

    for y in range(h):
        for x in range(w):
            z = height_map[y, x]
            vertices.append([x * scale, y * scale, z])

    def idx(x, y):
        return y * w + x

    for y in range(h - 1):
        for x in range(w - 1):
            v1 = idx(x, y)
            v2 = idx(x + 1, y)
            v3 = idx(x, y + 1)
            v4 = idx(x + 1, y + 1)

            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh


# ---------------------------
# MAIN GENERATOR
# ---------------------------
def generate_parametric_foot(mask, analysis):

    mask = preprocess_mask(mask)

    height_map = generate_height_map(mask, analysis)

    height_map = apply_heel_cup(height_map, mask)

    height_map = smooth_height_map(height_map)

    mesh = heightmap_to_mesh(height_map)

    return mesh