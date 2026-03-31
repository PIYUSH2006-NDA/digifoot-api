import cv2
import numpy as np
import os


def segment(image_path):
    print(f"🔍 Reading image: {image_path}")

    # --- Load image ---
    img = cv2.imread(image_path)

    # ❌ FIX: check if image loaded
    if img is None:
        raise Exception(f"❌ ERROR: Failed to load image -> {image_path}")

    # --- Resize ---
    try:
        img = cv2.resize(img, (256, 256))
    except Exception as e:
        raise Exception(f"❌ Resize failed for {image_path}: {str(e)}")

    # --- Convert to gray ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Threshold ---
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # --- Blur (reduce noise) ---
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # --- Morphology ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)

    if contours:
        # filter small noise
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean, [largest], -1, 255, -1)
        else:
            print("⚠️ No valid contour after filtering")
    else:
        print("⚠️ No contours found")

    # --- Final mask ---
    clean_mask = (clean > 0).astype("uint8")

    # ❌ EXTRA SAFETY: ensure mask is not empty
    if np.sum(clean_mask) == 0:
        raise Exception(f"❌ Empty mask generated for {image_path}")

    return clean_mask