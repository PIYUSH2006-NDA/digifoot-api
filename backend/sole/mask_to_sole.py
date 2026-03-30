import cv2
import numpy as np
import os
import trimesh
from scipy.spatial import Delaunay
from matplotlib.path import Path


def mask_to_sole(mask_path, output_dir):

    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise Exception("Mask not found")

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    contour = contour.squeeze()

    contour = contour.astype(np.float32)
    contour -= contour.min(axis=0)
    contour /= contour.max()

    grid = 120
    xs = np.linspace(0, 1, grid)
    ys = np.linspace(0, 1, grid)
    xv, yv = np.meshgrid(xs, ys)

    points = np.stack([xv.flatten(), yv.flatten()], axis=1)

    path = Path(contour)
    inside = path.contains_points(points)
    foot_points = points[inside]

    X = foot_points[:, 0]
    Y = foot_points[:, 1]

    Z = 6 * np.ones_like(X)

    arch = np.exp(-((X - 0.5)**2 / 0.02 + (Y - 0.6)**2 / 0.1))
    Z += 10 * arch

    heel = np.exp(-((Y - 0.1)**2 / 0.02))
    edge = np.abs(X - 0.5)
    Z += 5 * heel * edge

    Z += 2 * (1 - Y)

    vertices_top = np.column_stack([X * 220, Y * 100, Z])
    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = 0

    vertices = np.vstack([vertices_top, vertices_bottom])

    tri = Delaunay(foot_points)

    faces_top = tri.simplices
    faces_bottom = faces_top[:, ::-1] + len(vertices_top)

    edges = tri.convex_hull

    side_faces = []
    for e in edges:
        v1, v2 = e
        v1b = v1 + len(vertices_top)
        v2b = v2 + len(vertices_top)

        side_faces.append([v1, v2, v1b])
        side_faces.append([v2, v2b, v1b])

    faces = np.vstack([faces_top, faces_bottom, side_faces])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fill_holes()
    mesh.fix_normals()

    output_path = os.path.join(output_dir, "foot_sole.stl")
    mesh.export(output_path)

    print("✅ Orthotic STL generated:", output_path)

    return output_path