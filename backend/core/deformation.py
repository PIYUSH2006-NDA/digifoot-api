import numpy as np

def apply_arch_support(mesh, center_xy, delta_h, sigma_x=0.03, sigma_y=0.05):
    vertices = mesh.vertices

    for i, v in enumerate(vertices):
        dx = v[0] - center_xy[0]
        dy = v[1] - center_xy[1]

        gaussian = delta_h * np.exp(
            -((dx**2)/(2*sigma_x**2) + (dy**2)/(2*sigma_y**2))
        )

        vertices[i][2] += gaussian

    mesh.vertices = vertices
    return mesh


def apply_heel_and_forefoot(mesh, heel_center, m1, m5):
    vertices = mesh.vertices

    for i, v in enumerate(vertices):
        dx = v[0] - heel_center[0]
        dy = v[1] - heel_center[1]
        r = np.sqrt(dx**2 + dy**2)

        if r < 0.03:
            vertices[i][2] += 0.012

    toe_line = max(m1[1], m5[1])

    for i, v in enumerate(vertices):
        if v[1] > toe_line:
            vertices[i][2] = 0.002

    mesh.vertices = vertices
    return mesh