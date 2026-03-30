import numpy as np

def detect_keypoints(images, masks):
    keypoints = []

    for mask in masks:
        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        heel = (xs[ys.argmax()], ys.max())
        m1 = (xs.min(), ys.min())
        m5 = (xs.max(), ys.min())
        arch = (int(w * 0.3), int(h * 0.5))

        keypoints.append({
            "heel": heel,
            "m1": m1,
            "m5": m5,
            "arch": arch
        })

    return keypoints