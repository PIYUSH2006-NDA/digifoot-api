# deployment_coreml.py
# CoreML conversion + iOS optimization
# Run on macOS with: pip install coremltools ultralytics
# NOTE: CoreML conversion requires macOS

import numpy as np
from pathlib import Path


# ======================================================================
#  BASIC COREML EXPORT (via ultralytics)
# ======================================================================

def export_via_ultralytics(
    model_path: str,
    img_size: int = 640,
    use_half: bool = True,
    output_dir: str = ".",
):
    """
    Simplest export path: YOLO → ONNX → CoreML.
    ultralytics handles the full chain.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    path = model.export(
        format="coreml",
        imgsz=img_size,
        half=use_half,           # FP16 → Apple Neural Engine
        nms=True,                # embed NMS into model graph
        simplify=True,
        opset=16,
    )
    print(f"Exported: {path}")
    return path


# ======================================================================
#  ADVANCED EXPORT WITH INT8 WEIGHT QUANTIZATION
# ======================================================================

def export_quantized(
    onnx_path: str,
    output_path: str = "FootSeg_quantized.mlpackage",
    img_size: int = 640,
):
    """
    ONNX → CoreML with INT8 weight quantization.
    Reduces model size ~4x, maintains FP16 activations.
    Best for deployment on A-series / M-series chips.
    Requires: coremltools >= 7.0
    """
    try:
        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linearly_quantize_weights,
        )
    except ImportError:
        print("Install: pip install coremltools>=7.0")
        return

    print("Converting ONNX → CoreML...")
    mlmodel = ct.convert(
        onnx_path,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, img_size, img_size),
                color_layout=ct.colorlayout.BGR,
                scale=1 / 255.0,
                bias=[0, 0, 0],
            )
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,    # ANE + GPU + CPU
    )

    print("Applying INT8 weight quantization...")
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        weight_threshold=512,   # quantize weights with >512 elements
    )
    config = OptimizationConfig(global_config=op_config)
    compressed = linearly_quantize_weights(mlmodel, config=config)

    # Metadata
    compressed.short_description = "Foot segmentation from TrueDepth/LiDAR depth maps"
    compressed.author = "FootScanSystem"
    compressed.version = "1.0"
    compressed.input_description["image"] = (
        "3-channel pseudo-RGB: Ch0=depth, Ch1=surface normals, Ch2=curvature"
    )

    compressed.save(output_path)
    print(f"Saved quantized CoreML: {output_path}")

    # Print size
    import os
    size_mb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"Model size: {size_mb:.1f} MB")
    return output_path


# ======================================================================
#  BENCHMARK
# ======================================================================

def benchmark_coreml(
    mlpackage_path: str,
    n_runs: int = 50,
    img_size: int = 640,
):
    """
    Benchmark CoreML inference latency on current machine.
    NOTE: Run on target device (iPhone/Mac) for real numbers.
    """
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not available")
        return

    import time

    model = ct.models.MLModel(mlpackage_path)
    dummy = {"image": np.random.rand(1, 3, img_size, img_size).astype(np.float32)}

    # Warm up
    for _ in range(5):
        model.predict(dummy)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy)
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    print(f"\nCoreML Benchmark ({n_runs} runs)")
    print(f"  Mean:   {np.mean(times_ms):.1f} ms")
    print(f"  Median: {np.median(times_ms):.1f} ms")
    print(f"  Min:    {np.min(times_ms):.1f} ms")
    print(f"  Max:    {np.max(times_ms):.1f} ms")
    print(f"  Est. FPS: {1000/np.mean(times_ms):.1f}")


# ======================================================================
#  iOS SWIFT INTEGRATION GUIDE (as string comments)
# ======================================================================

SWIFT_INTEGRATION_GUIDE = """
// ============================================================
// iOS Swift Integration — FootScanProcessor.swift
// ============================================================
//
// 1. Add FootSeg.mlpackage to Xcode project
// 2. Add ARKit + CoreML imports
// 3. Use ARKit sceneDepth for LiDAR or TrueDepth depthMap
//
// import ARKit
// import CoreML
// import Vision
// import Accelerate
//
// class FootScanProcessor: NSObject, ARSessionDelegate {
//
//     let model: FootSeg       // auto-generated from .mlpackage
//     let trigger: ScanTrigger // your Swift trigger class
//     var isCapturing = false
//
//     func session(_ session: ARSession, didUpdate frame: ARFrame) {
//         guard !isCapturing else { return }
//
//         // Get depth from LiDAR (iOS 15.4+) or TrueDepth
//         guard let depth = frame.sceneDepth?.depthMap
//               ?? frame.capturedDepthData?.depthDataMap else { return }
//
//         // Convert CVPixelBuffer → [Float32]
//         let depthArray = extractFloat32Array(from: depth)
//
//         // Build 3-channel pseudo-RGB input
//         let inputPixelBuffer = buildPseudoRGB(depth: depthArray)
//
//         // Run CoreML
//         guard let output = try? model.prediction(image: inputPixelBuffer) else { return }
//
//         // Extract mask + confidence from output
//         let (mask, conf) = parsePrediction(output)
//
//         // Update scan trigger
//         let state = trigger.update(depth: depthArray, mask: mask, conf: conf)
//
//         // Update UI on main thread
//         DispatchQueue.main.async {
//             self.updateUI(state: state)
//         }
//
//         // Capture when ready
//         if state.readyToCapture && !isCapturing {
//             isCapturing = true
//             captureAndReconstruct(depth: depthArray, mask: mask)
//         }
//     }
//
//     func buildPseudoRGB(depth: [Float]) -> CVPixelBuffer {
//         // Ch0: normalize depth to [0,255]
//         // Ch1: Sobel gradient magnitude (surface normals)
//         // Ch2: Laplacian magnitude (curvature)
//         // Pack into BGRA CVPixelBuffer, ignore A channel
//         // ... (use vDSP for fast SIMD operations)
//     }
//
//     func captureAndReconstruct(depth: [Float], mask: [UInt8]) {
//         // 1. Apply mask to depth
//         // 2. Send to reconstruction service (server or on-device)
//         // 3. On-device: use Metal for point cloud generation
//         // 4. Send .obj or .ply to backend for mesh processing
//         trigger.reset()
//         isCapturing = false
//     }
// }
//
// ============================================================
// Performance targets on Apple Silicon:
//   YOLOv8n-seg FP16 + ANE: ~15–30ms → 30–60 FPS
//   Preprocessing (vDSP):   ~5ms
//   Floor removal:           ~3ms
//   Total pipeline:          ~25–40ms → 25–40 FPS
// ============================================================
"""


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoreML export tools")
    sub = parser.add_subparsers(dest="cmd")

    p_exp = sub.add_parser("export", help="Export PyTorch → CoreML")
    p_exp.add_argument("--model", required=True, help=".pt model path")
    p_exp.add_argument("--method", default="ultralytics",
                       choices=["ultralytics", "quantized"])
    p_exp.add_argument("--onnx", help="ONNX path (for --method quantized)")

    p_bench = sub.add_parser("benchmark", help="Benchmark CoreML model")
    p_bench.add_argument("--model", required=True, help=".mlpackage path")

    args = parser.parse_args()

    if args.cmd == "export":
        if args.method == "ultralytics":
            export_via_ultralytics(args.model)
        elif args.method == "quantized":
            assert args.onnx, "--onnx required for quantized method"
            export_quantized(args.onnx)

    elif args.cmd == "benchmark":
        benchmark_coreml(args.model)

    else:
        # Print Swift guide
        print(SWIFT_INTEGRATION_GUIDE)
