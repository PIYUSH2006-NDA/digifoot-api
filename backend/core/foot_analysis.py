from backend.core.arch import calculate_arch_index, classify_arch, get_arch_height
from backend.core.alignment import calculate_heel_alignment, get_alignment_correction
from backend.core.validation import validate_mask


def analyze_foot(mask):
    """
    Main analysis engine
    """

    # Validate
    valid, msg = validate_mask(mask)
    if not valid:
        return {"error": msg}

    # Arch
    arch_index = calculate_arch_index(mask)
    arch_type = classify_arch(arch_index)
    arch_height = get_arch_height(arch_type)

    # Heel alignment
    heel_imbalance = calculate_heel_alignment(mask)
    correction = get_alignment_correction(heel_imbalance)

    return {
        "arch_index": float(arch_index),
        "arch_type": arch_type,
        "arch_height": arch_height,
        "heel_imbalance": float(heel_imbalance),
        "heel_correction": float(correction)
    }