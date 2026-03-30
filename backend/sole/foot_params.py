import cv2
import numpy as np

def extract_params(image_paths):
    bottom = cv2.imread(image_paths[0], 0)
    side = cv2.imread(image_paths[1], 0)

    if bottom is None or side is None:
        raise ValueError("Error loading images")

    # ---------------------------
    # FOOT LENGTH & WIDTH
    # ---------------------------
    _, thresh = cv2.threshold(bottom, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # Normalize (important for Blender scaling)
        foot_length = float(h)
        foot_width = float(w)
    else:
        foot_length, foot_width = 100.0, 40.0

    # ---------------------------
    # ARCH HEIGHT (Better detection)
    # ---------------------------
    edges = cv2.Canny(side, 50, 150)

    heights = []
    for col in range(edges.shape[1]):
        ys = np.where(edges[:, col] > 0)[0]
        if len(ys) > 0:
            heights.append(edges.shape[0] - ys.min())

    if len(heights) == 0:
        arch_height = 0.2
    else:
        arch_height = np.clip(np.mean(heights) / edges.shape[0], 0.1, 0.5)

def extract_insole_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return contour

def normalize_scale(mask, real_width_mm=210):
    h, w = mask.shape
    pixels_per_mm = w / real_width_mm
    return pixels_per_mm

def extract_advanced_params(contour, scale):
    x, y, w, h = cv2.boundingRect(contour)

    length = h / scale
    width = w / scale

    heel_width = width * 0.6
    arch_width = width * 0.4
    forefoot_width = width * 0.8

    return {
        "length": length,
        "width": width,
        "heel_width": heel_width,
        "arch_width": arch_width,
        "forefoot_width": forefoot_width
    }
