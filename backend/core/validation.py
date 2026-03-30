import numpy as np


def validate_mask(mask):
    """
    Ensure mask is usable
    """

    if mask is None:
        return False, "Mask is None"

    if np.sum(mask) < 1000:
        return False, "Foot not detected properly"

    return True, "Valid"