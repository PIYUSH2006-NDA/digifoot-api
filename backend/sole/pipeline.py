import os
import cv2
import time
import hashlib
import numpy as np
import subprocess
import json

from backend.sole.foot_params import extract_params
from inference import segment
from backend.core.biomechanics import apply_biomechanical_corrections
from backend.ml.model import load_model, predict_depth
from backend.ml.multiview import reconstruct_3d
from backend.core.materials import apply_material_zones
from backend.core.realistic_shape import apply_realistic_shape
from backend.core.foot_analysis import analyze_foot
from backend.ml.parametric import generate_parametric_foot
from backend.core.optimization import optimize_mesh, smooth_mesh, normalize_for_ar
import trimesh


# ---------------------------
# 🔥 NEW: ADVANCED EXTRACTION
# ---------------------------
def extract_insole_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No contour found")
    return max(contours, key=cv2.contourArea)


def normalize_scale(mask, real_width_mm=210):
    h, w = mask.shape
    pixels_per_mm = w / real_width_mm
    return pixels_per_mm


def extract_advanced_params(contour, scale):
    x, y, w, h = cv2.boundingRect(contour)

    length = h / scale
    width = w / scale

    return {
        "length": float(length),
        "width": float(width),
        "heel_width": float(width * 0.6),
        "arch_width": float(width * 0.4),
        "forefoot_width": float(width * 0.8)
    }


def apply_corrections(params, foot_type):
    if foot_type == "flat":
        params["arch_height"] = 12
    elif foot_type == "high":
        params["arch_height"] = 6
    else:
        params["arch_height"] = 8

    params["heel_cup"] = 5
    params["thickness"] = 4

    return params


# Load model once
model = load_model()
try:
    model.eval()
except:
    pass


def classify_foot(mask):
    h, w = mask.shape
    arch_region = mask[:, int(w * 0.4):int(w * 0.6)]
    arch_density = arch_region.sum() / arch_region.size

    if arch_density > 0.6:
        return "flat"
    elif arch_density > 0.3:
        return "normal"
    else:
        return "high"


def get_cache_key(image_paths):
    key = "".join(image_paths)
    return hashlib.md5(key.encode()).hexdigest()


def run_pipeline(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cache_key = get_cache_key(image_paths)
    cache_path = os.path.abspath(os.path.join(output_dir, f"{cache_key}.stl"))

    masks = []

    # ---------------------------
    # STEP 1: SEGMENT
    # ---------------------------
    for path in image_paths:
        mask = segment(path)
        if mask is None:
            raise Exception(f"Segmentation failed: {path}")
        masks.append(mask)

    # ---------------------------
    # STEP 2: COMBINE
    # ---------------------------
    combined = np.mean(masks, axis=0)
    combined = (combined > 0.5).astype("uint8")

    combined = cv2.GaussianBlur(combined.astype(np.float32), (15, 15), 0)
    combined = (combined > 0.25).astype("uint8")

    # ---------------------------
    # 🔥 ML + MULTIVIEW DEPTH
    # ---------------------------
    try:
        print("🧠 ML Depth Prediction...")
        depth_map = predict_depth(model, combined)
    except Exception as e:
        print("⚠️ ML failed, fallback to multiview:", str(e))
        depth_map = reconstruct_3d(masks)

    # ---------------------------
    # FINAL MASK
    # ---------------------------
    combined = (depth_map > 0.2).astype("uint8")
    combined = cv2.GaussianBlur(combined.astype(np.float32), (15, 15), 0)
    combined = (combined > 0.25).astype("uint8")

    final_mask_path = os.path.abspath(os.path.join(output_dir, "final_mask.png"))
    cv2.imwrite(final_mask_path, (combined * 255).astype("uint8"))

    # ---------------------------
    # 🧠 ANALYSIS
    # ---------------------------
    analysis = analyze_foot(combined)
    if "error" in analysis:
        raise Exception(analysis["error"])

    print("🧠 Foot Analysis:", analysis)

    # ---------------------------
    # 🔥 NEW: ADVANCED PARAM FLOW
    # ---------------------------
    contour = extract_insole_shape(combined)
    scale = normalize_scale(combined)
    advanced_params = extract_advanced_params(contour, scale)

    base_params = extract_params(image_paths)

    params = {
        **(base_params or {}),
        **(advanced_params or {})
    }

    # ---------------------------
    # FOOT TYPE + CORRECTION
    # ---------------------------
    foot_type = classify_foot(combined)
    params = apply_corrections(params, foot_type)

    # ---------------------------
    # 🔥 PARAMETRIC MESH
    # ---------------------------
    try:
        print("🧠 Generating AI parametric mesh...")

        param_mesh = generate_parametric_foot(depth_map, analysis)
        param_mesh = apply_biomechanical_corrections(param_mesh, analysis)
        param_mesh = apply_material_zones(param_mesh, analysis)
        param_mesh = apply_realistic_shape(param_mesh)

        param_mesh = optimize_mesh(param_mesh)
        param_mesh = smooth_mesh(param_mesh)
        param_mesh = normalize_for_ar(param_mesh)

        param_stl = os.path.abspath(os.path.join(output_dir, "parametric_insole.stl"))
        param_mesh.export(param_stl)

        print("✅ Parametric STL GENERATED")

    except Exception as e:
        print("⚠️ Parametric generation failed:", str(e))
        param_stl = None

    # ---------------------------
    # STEP 4: BLENDER (FIXED)
    # ---------------------------
    blender_path = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

    output_stl = os.path.join(output_dir, f"foot_sole_{int(time.time())}.stl")
    deform_script = os.path.abspath(os.path.join("backend", "sole", "deform_sole.py"))

    param_json = json.dumps(params)

    # 🔥 DEBUG
    print("\n🚀 Blender Execution Debug:")
    print("Mask Path:", final_mask_path)
    print("Output STL:", output_stl)
    print("Params JSON (short):", param_json[:200])
    print("Foot Type:", foot_type)

    command = [
        blender_path,
        "--background",
        "--python", deform_script,
        "--",
        str(final_mask_path),
        str(output_stl),
        str(param_json),
        str(foot_type)
    ]

    print("🚀 Running Blender...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )

    print("\n--- Blender STDOUT ---")
    print(result.stdout)

    print("\n--- Blender STDERR ---")
    print(result.stderr)

    if result.returncode != 0:
        raise Exception(f"Blender failed with code {result.returncode}")

    if not os.path.exists(output_stl):
        raise Exception("STL file not generated")

    print("✅ Blender STL GENERATED")

    # ---------------------------
    # CACHE SAVE
    # ---------------------------
    try:
        import shutil
        shutil.copy(output_stl, cache_path)
    except:
        pass

    return {
        "blender_stl": output_stl,
        "parametric_stl": param_stl,
        "analysis": analysis,
        "params": params
    }