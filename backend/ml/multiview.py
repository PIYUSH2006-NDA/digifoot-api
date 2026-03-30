import numpy as np
import cv2


# ---------------------------
# ALIGN IMAGES (CENTER + SCALE)
# ---------------------------
def align_masks(masks):
    aligned = []

    target_size = (256, 256)

    for mask in masks:
        mask = cv2.resize(mask, target_size)

        # Find bounding box
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            aligned.append(mask)
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = mask[y_min:y_max, x_min:x_max]

        # Resize to standard
        cropped = cv2.resize(cropped, target_size)

        aligned.append(cropped)

    return aligned


# ---------------------------
# GENERATE DEPTH FROM MULTI VIEW
# ---------------------------
def generate_depth_map(masks):
    h, w = masks[0].shape

    depth = np.zeros((h, w), dtype=np.float32)

    for i, mask in enumerate(masks):
        weight = (i + 1) / len(masks)
        depth += mask.astype(np.float32) * weight

    depth /= len(masks)

    return depth


# ---------------------------
# ADD CURVATURE (FOOT SHAPE)
# ---------------------------
def apply_foot_curvature(depth):
    h, w = depth.shape

    for y in range(h):
        for x in range(w):
            curve = np.sin(np.pi * x / w) * 0.3
            depth[y, x] += curve

    return depth


# ---------------------------
# SMOOTH DEPTH
# ---------------------------
def smooth_depth(depth):
    return cv2.GaussianBlur(depth, (11, 11), 0)


# ---------------------------
# MAIN MULTI VIEW
# ---------------------------
def reconstruct_3d(masks):

    masks = align_masks(masks)

    depth = generate_depth_map(masks)

    depth = apply_foot_curvature(depth)

    depth = smooth_depth(depth)

    return depth