import cv2
import numpy as np

def segment_foot(images):
    masks = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold (works for foot images)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (foot should be white)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        # Clean noise
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        mask = (thresh > 0).astype(np.uint8)

        masks.append(mask)

    return masks