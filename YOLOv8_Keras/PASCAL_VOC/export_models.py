"""
Export the trained YOLOv8 Keras inference model.

Examples
--------
    python export_models.py --formats saved_model tflite
    python export_models.py --formats saved_model tflite tftrt
    python export_models.py --formats all --fp16

Notes
-----
TF-TRT and TensorRT engine export require NVIDIA TensorRT support. They will not
work on macOS/CPU-only TensorFlow installs.
"""

import argparse
import os
import shutil
import subprocess
import sys

import tensorflow as tf

from voc_cfg import TRAIN_CFG
from yolov8_architecture.yolov8_architecture import build_yolov8


DEFAULT_VARIANT = "n"
DEFAULT_NUM_CLASSES = 8
DEFAULT_RUN_DIR = f"runs/yolov8{DEFAULT_VARIANT}_voc"
DEFAULT_WEIGHTS = os.path.join(DEFAULT_RUN_DIR, "best_inference.weights.h5")
DEFAULT_EXPORT_DIR = os.path.join(DEFAULT_RUN_DIR, "export")


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 Keras model.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--imgsz", type=int, default=TRAIN_CFG["imgsz"])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["saved_model", "tflite"],
        choices=["saved_model", "tflite", "tftrt", "onnx", "trt", "all"],
        help="'trt' exports ONNX first, then calls trtexec when available.",
    )
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for TFLite/TF-TRT/TRT when supported.")
    parser.add_argument("--int8", action="store_true",
                        help="Create dynamic-range INT8 TFLite. Calibration is not included.")
    parser.add_argument("--trt-workspace", type=int, default=4096,
                        help="TensorRT workspace size in MiB for trtexec.")
    return parser.parse_args()


def normalized_formats(formats):
    requested = set(formats)
    if "all" in requested:
        return {"saved_model", "tflite", "tftrt", "onnx", "trt"}
    if "trt" in requested:
        requested.add("onnx")
    if "tftrt" in requested or "onnx" in requested:
        requested.add("saved_model")
    return requested


def build_and_load(args):
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = build_yolov8(
        variant=args.variant,
        task="detect",
        input_shape=args.imgsz,
        num_classes=args.num_classes,
        training=False,
        print_summary=False,
    )
    model.load_weights(args.weights)

    dummy = tf.zeros([1, args.imgsz, args.imgsz, 3], dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


def export_saved_model(model, args):
    path = os.path.join(args.out, "saved_model")
    os.makedirs(args.out, exist_ok=True)
    model.save(path, include_optimizer=False)
    print(f"SavedModel -> {path}")
    return path


def export_tflite(model, args):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        suffix = "fp16"
    elif args.int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        suffix = "int8_dynamic"
    else:
        suffix = "fp32"

    tflite_model = converter.convert()
    path = os.path.join(args.out, f"model_{suffix}.tflite")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite -> {path}")
    return path


def export_tftrt(saved_model_dir, args):
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except Exception as exc:
        print(f"TF-TRT unavailable in this TensorFlow install: {exc}")
        return None

    precision = "FP16" if args.fp16 else "FP32"
    params = trt.TrtConversionParams(
        precision_mode=precision,
        max_workspace_size_bytes=args.trt_workspace * 1024 * 1024,
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params,
    )
    converter.convert()

    def input_fn():
        yield (tf.zeros([1, args.imgsz, args.imgsz, 3], dtype=tf.float32),)

    converter.build(input_fn=input_fn)
    path = os.path.join(args.out, f"tftrt_{precision.lower()}_saved_model")
    converter.save(path)
    print(f"TF-TRT SavedModel -> {path}")
    return path


def export_onnx(saved_model_dir, args):
    path = os.path.join(args.out, "model.onnx")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", path,
        "--opset", "13",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ONNX export failed. Install tf2onnx first if needed: {exc}")
        return None
    print(f"ONNX -> {path}")
    return path


def export_trt_engine(onnx_path, args):
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print("TensorRT engine export skipped: trtexec was not found in PATH.")
        return None

    engine_path = os.path.join(args.out, "model_fp16.engine" if args.fp16 else "model_fp32.engine")
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.trt_workspace}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine -> {engine_path}")
    return engine_path


def main():
    args = parse_args()
    formats = normalized_formats(args.formats)

    os.makedirs(args.out, exist_ok=True)
    model = build_and_load(args)

    saved_model_dir = None
    if "saved_model" in formats:
        saved_model_dir = export_saved_model(model, args)

    if "tflite" in formats:
        export_tflite(model, args)

    if "tftrt" in formats:
        saved_model_dir = saved_model_dir or export_saved_model(model, args)
        export_tftrt(saved_model_dir, args)

    onnx_path = None
    if "onnx" in formats:
        saved_model_dir = saved_model_dir or export_saved_model(model, args)
        onnx_path = export_onnx(saved_model_dir, args)

    if "trt" in formats:
        if onnx_path is not None:
            export_trt_engine(onnx_path, args)
        else:
            print("TensorRT engine export skipped because ONNX export did not complete.")


if __name__ == "__main__":
    main()
