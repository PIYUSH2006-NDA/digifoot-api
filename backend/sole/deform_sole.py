import bpy
import sys
import os
import math
import struct
import json

# ---------------------------
# ARGUMENTS (UPDATED SAFE)
# ---------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:]

# 🔥 DEFAULTS (fallback safety)
length = 250.0
width = 100.0
arch_val = 8.0
foot_type = "normal"

# ---------------------------
# NEW JSON MODE (PRIMARY)
# ---------------------------
if len(argv) == 4:
    mask_path = os.path.abspath(argv[0])
    output_path = os.path.abspath(argv[1])

    try:
        params = json.loads(argv[2])
    except Exception as e:
        print("JSON parse error:", e)
        params = {}

    foot_type = argv[3]

    # Extract values safely
    length = float(params.get("length", length))
    width = float(params.get("width", width))
    arch_val = float(params.get("arch_height", arch_val))

    # 🔥 Dummy walls (reuse mask if not provided)
    walls_path = mask_path

# ---------------------------
# OLD MODE (BACKWARD COMPATIBLE)
# ---------------------------
else:
    mask_path = os.path.abspath(argv[0])
    walls_path = os.path.abspath(argv[1])
    output_path = os.path.abspath(argv[2])
    length = float(argv[3])
    width = float(argv[4])
    arch_val = float(argv[5])
    foot_type = argv[6]

# ---------------------------
# DEBUG
# ---------------------------
print("\n--- Blender Input Debug ---")
print("Mask:", mask_path)
print("Walls:", walls_path)
print("Output:", output_path)
print("Length:", length)
print("Width:", width)
print("Arch:", arch_val)
print("Foot Type:", foot_type)

# ---------------------------
# CLEAR SCENE
# ---------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ---------------------------
# LOAD IMAGES
# ---------------------------
img_mask = bpy.data.images.load(mask_path)
img_walls = bpy.data.images.load(walls_path)

img_width, img_height = img_mask.size
mask_pixels = list(img_mask.pixels)
wall_pixels = list(img_walls.pixels)

# ---------------------------
# GRID
# ---------------------------
bpy.ops.mesh.primitive_grid_add(x_subdivisions=300, y_subdivisions=600, size=2)
obj = bpy.context.active_object
mesh = obj.data

verts_to_delete = []

# ---------------------------
# SHAPE GENERATION
# ---------------------------
for v in mesh.vertices:
    nx = (v.co.x + 1) * 0.5
    ny = (v.co.y + 1) * 0.5

    x = int(nx * (img_width - 1))
    y = int(ny * (img_height - 1))

    idx = (y * img_width + x) * 4

    mask_val = mask_pixels[idx]

    if mask_val < 0.2:
        verts_to_delete.append(v.index)
        continue

    wall_val = wall_pixels[idx]
    cup = wall_val ** 2

    z = cup * 0.12

    # Heel
    if ny < 0.25:
        z += 0.08 * (1 - ny / 0.25)

    # Arch
    if nx < 0.5:
        arch_zone = math.exp(-((nx - 0.25)**2)*20 - ((ny - 0.5)**2)*8)

        if foot_type == "flat":
            z += arch_zone * 0.05
        elif foot_type == "normal":
            z += arch_zone * 0.10
        else:
            z += arch_zone * 0.18

    # Toe
    if ny > 0.8:
        z += 0.05 * ((ny - 0.8) / 0.2)

    v.co.z = z

# ---------------------------
# DELETE UNUSED
# ---------------------------
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

for idx in verts_to_delete:
    mesh.vertices[idx].select = True

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.delete(type='VERT')
bpy.ops.object.mode_set(mode='OBJECT')

# ---------------------------
# SCALE
# ---------------------------
obj.scale[0] = width / 1000
obj.scale[1] = length / 1000

# ---------------------------
# MODIFIERS
# ---------------------------
smooth = obj.modifiers.new("smooth", 'SMOOTH')
smooth.iterations = 20

solid = obj.modifiers.new("solid", 'SOLIDIFY')
solid.thickness = 0.01
solid.offset = 1

sub = obj.modifiers.new("subd", 'SUBSURF')
sub.levels = 2

dec = obj.modifiers.new("dec", 'DECIMATE')
dec.ratio = 0.4

bpy.ops.object.shade_smooth()

# APPLY MODIFIERS
bpy.context.view_layer.objects.active = obj
for mod in obj.modifiers:
    bpy.ops.object.modifier_apply(modifier=mod.name)

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.polygons))

# ---------------------------
# STL WRITER
# ---------------------------
def write_stl(obj, filepath):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    if len(mesh.polygons) == 0:
        raise Exception("Mesh has no faces!")

    with open(filepath, 'wb') as f:
        f.write(b'Python STL Writer' + b' ' * (80 - len('Python STL Writer')))
        f.write(struct.pack('<I', len(mesh.polygons)))

        for poly in mesh.polygons:
            normal = poly.normal
            f.write(struct.pack('<3f', normal.x, normal.y, normal.z))

            for idx in poly.vertices:
                v = mesh.vertices[idx].co
                f.write(struct.pack('<3f', v.x, v.y, v.z))

            f.write(struct.pack('<H', 0))

    eval_obj.to_mesh_clear()

# ---------------------------
# EXPORT
# ---------------------------
write_stl(obj, output_path)

print("✅ STL EXPORTED (DIRECT METHOD)")