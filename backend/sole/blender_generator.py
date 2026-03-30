import bpy
import sys
import os
import math

# ---------------------------
# GET ARGUMENTS
# ---------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:]

mask_path = os.path.abspath(argv[0])
output_path = os.path.abspath(argv[1])
foot_type = argv[2]

print("Foot type:", foot_type)
print("Mask:", mask_path)

# ---------------------------
# CLEAR SCENE
# ---------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ---------------------------
# LOAD IMAGE (SAFE METHOD)
# ---------------------------
img = bpy.data.images.load(mask_path)
width, height = img.size
pixels = list(img.pixels)

# ---------------------------
# GRID (OPTIMIZED)
# ---------------------------
bpy.ops.mesh.primitive_grid_add(
    x_subdivisions=120,
    y_subdivisions=60,
    size=2
)

obj = bpy.context.active_object
mesh = obj.data

obj.scale[0] = 200
obj.scale[1] = 100

# ---------------------------
# HEIGHT MAP (FIXED + STRONG)
# ---------------------------
for v in mesh.vertices:

    nx = (v.co.x + 1) * 0.5
    ny = (v.co.y + 1) * 0.5

    x = int(nx * (width - 1))
    y = int(ny * (height - 1))

    if x < 0 or y < 0 or x >= width or y >= height:
        v.co.z = 0
        continue

    idx = (y * width + x) * 4

    # RGB average (important fix)
    r = pixels[idx]
    g = pixels[idx + 1]
    b = pixels[idx + 2]

    val = (r + g + b) / 3.0

    if val < 0.2:
        v.co.z = 0
    else:
        # ---------------------------
        # BASE HEIGHT (STRONGER)
        # ---------------------------
        z = val * 8

        # ---------------------------
        # ARCH SUPPORT
        # ---------------------------
        arch = math.exp(-((nx - 0.55)**2)/0.02)

        if foot_type == "flat":
            z += arch * 2
        elif foot_type == "normal":
            z += arch * 5
        else:
            z += arch * 8

        # ---------------------------
        # HEEL CUP
        # ---------------------------
        heel = math.exp(-((nx - 0.2)**2 + (ny - 0.5)**2)/0.03)
        z += heel * 6

        # ---------------------------
        # TOE SPRING
        # ---------------------------
        if ny > 0.75:
            z += (ny - 0.75)**2 * 8

        # ---------------------------
        # EDGE CURVATURE
        # ---------------------------
        edge = abs(nx - 0.5)
        z += edge**2 * 4

        v.co.z = z * 0.02


# ---------------------------
# THICKNESS
# ---------------------------
solid = obj.modifiers.new(name="solid", type='SOLIDIFY')
solid.thickness = 0.025
solid.offset = 1

# ---------------------------
# SMOOTHING
# ---------------------------
sub = obj.modifiers.new(name="subd", type='SUBSURF')
sub.levels = 2

bpy.ops.object.shade_smooth()

# ---------------------------
# DECIMATE (SIZE CONTROL)
# ---------------------------
dec = obj.modifiers.new(name="decimate", type='DECIMATE')
dec.ratio = 0.3

# ---------------------------
# APPLY MODIFIERS
# ---------------------------
bpy.context.view_layer.objects.active = obj

for mod in obj.modifiers:
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except:
        pass

# ---------------------------
# CLEAN MESH
# ---------------------------
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.remove_doubles()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# ---------------------------
# EXPORT STL (SAFE)
# ---------------------------
try:
    bpy.ops.export_mesh.stl(
        filepath=output_path,
        use_selection=True,
        ascii=False
    )
except:
    bpy.ops.wm.stl_export(filepath=output_path)

print("✅ FINAL SOLE GENERATED SUCCESSFULLY")