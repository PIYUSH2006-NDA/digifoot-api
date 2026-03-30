import numpy as np


def apply_biomechanical_corrections(mesh, analysis):
    """
    Modify mesh vertices based on foot biomechanics
    """

    vertices = mesh.vertices.copy()

    arch_height = analysis.get("arch_height", 0.5)
    foot_type = analysis.get("foot_type", "normal")
    pressure_map = analysis.get("pressure_map", None)

    print("🦶 Applying biomechanical corrections...")

    # ---------------------------
    # 1. ARCH SUPPORT
    # ---------------------------
    arch_strength = get_arch_strength(foot_type, arch_height)

    vertices = apply_arch_support(vertices, arch_strength)

    # ---------------------------
    # 2. HEEL CUP
    # ---------------------------
    vertices = apply_heel_cup(vertices)

    # ---------------------------
    # 3. PRESSURE RELIEF
    # ---------------------------
    if pressure_map is not None:
        vertices = apply_pressure_relief(vertices, pressure_map)

    # ---------------------------
    # 4. PRONATION / SUPINATION FIX
    # ---------------------------
    vertices = apply_pronation_control(vertices, foot_type)

    mesh.vertices = vertices

    print("✅ Biomechanical corrections applied")

    return mesh


# ---------------------------
# ARCH SUPPORT
# ---------------------------
def get_arch_strength(foot_type, arch_height):

    if foot_type == "flat":
        return 1.5
    elif foot_type == "high":
        return 0.6
    else:
        return 1.0


def apply_arch_support(vertices, strength):

    for i, v in enumerate(vertices):
        x, y, z = v

        # middle of foot
        if 0.3 < x < 0.7:
            lift = strength * (1 - abs(0.5 - x)) * 0.02
            vertices[i][2] += lift

    return vertices


# ---------------------------
# HEEL CUP
# ---------------------------
def apply_heel_cup(vertices):

    for i, v in enumerate(vertices):
        x, y, z = v

        # back region
        if x < 0.2:
            lift = (0.2 - x) * 0.03
            vertices[i][2] += lift

    return vertices


# ---------------------------
# PRESSURE RELIEF
# ---------------------------
def apply_pressure_relief(vertices, pressure_map):

    for i, v in enumerate(vertices):
        x, y, z = v

        px = int(x * pressure_map.shape[1])
        py = int(y * pressure_map.shape[0])

        if px < pressure_map.shape[1] and py < pressure_map.shape[0]:
            pressure = pressure_map[py][px]

            # reduce height in high pressure zones
            vertices[i][2] -= pressure * 0.01

    return vertices


# ---------------------------
# PRONATION CONTROL
# ---------------------------
def apply_pronation_control(vertices, foot_type):

    for i, v in enumerate(vertices):
        x, y, z = v

        # medial side correction
        if foot_type == "flat":
            if y < 0.5:
                vertices[i][2] += 0.01

        # lateral correction
        elif foot_type == "high":
            if y > 0.5:
                vertices[i][2] += 0.008

    return vertices