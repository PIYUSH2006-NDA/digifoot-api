import trimesh


# ---------------------------
# MESH OPTIMIZATION
# ---------------------------
def optimize_mesh(mesh, target_faces=800):
    print("⚙️ Optimizing mesh...")

    try:
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        print(f"✅ Reduced faces: {len(mesh.faces)} → {len(simplified.faces)}")
        return simplified
    except Exception as e:
        print("⚠️ Simplification failed:", str(e))
        return mesh


# ---------------------------
# SMOOTH MESH
# ---------------------------
def smooth_mesh(mesh):
    print("✨ Smoothing mesh...")

    try:
        mesh = mesh.smoothed()
        print("✅ Smooth complete")
        return mesh
    except Exception as e:
        print("⚠️ Smooth failed:", str(e))
        return mesh


# ---------------------------
# AR NORMALIZATION
# ---------------------------
def normalize_for_ar(mesh):
    print("📐 Normalizing for AR...")

    try:
        # Center mesh
        mesh.vertices -= mesh.center_mass

        # Scale uniformly
        scale = 1.0 / max(mesh.extents)
        mesh.vertices *= scale

        print("✅ AR normalization done")
        return mesh

    except Exception as e:
        print("⚠️ AR normalization failed:", str(e))
        return mesh