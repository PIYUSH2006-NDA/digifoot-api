import numpy as np


# ---------------------------
# ARCH INDEX CALCULATION
# ---------------------------
def calculate_arch_index(mask):
    """
    Arch Index = midfoot area / total foot area
    """

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    # Divide foot into 3 parts (heel, midfoot, forefoot)
    third = h // 3

    heel = mask[0:third, :]
    midfoot = mask[third:2*third, :]
    forefoot = mask[2*third:h, :]

    total_area = np.sum(mask)
    mid_area = np.sum(midfoot)

    if total_area == 0:
        return 0

    arch_index = mid_area / total_area

    return arch_index


# ---------------------------
# ARCH CLASSIFICATION
# ---------------------------
def classify_arch(arch_index):
    """
    Returns: flat / normal / high
    """

    if arch_index > 0.55:
        return "flat"
    elif arch_index > 0.40:
        return "normal"
    else:
        return "high"


# ---------------------------
# ARCH HEIGHT (mm)
# ---------------------------
def get_arch_height(arch_type):
    """
    Real-world approximate values
    """

    if arch_type == "flat":
        return 8.0   # more support
    elif arch_type == "normal":
        return 5.0
    else:
        return 3.0   # high arch needs less lift