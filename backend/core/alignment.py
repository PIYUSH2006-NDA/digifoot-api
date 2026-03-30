import numpy as np


def calculate_heel_alignment(mask):
    """
    Detect heel tilt (left/right imbalance)
    """

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    heel_region = mask[0:int(h*0.2), :]

    left = np.sum(heel_region[:, :w//2])
    right = np.sum(heel_region[:, w//2:])

    if left + right == 0:
        return 0

    imbalance = (left - right) / (left + right)

    return imbalance  # negative → right tilt, positive → left tilt


def get_alignment_correction(imbalance):
    """
    Returns correction factor
    """

    return -imbalance * 5.0  # scale factor