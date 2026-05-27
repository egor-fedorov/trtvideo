#!/usr/bin/env python3
"""Export RealESRGAN (.pth) to ONNX with fixed input dimensions.

Creates ONNX files for 720p and 1080p resolutions.

Usage:
    export-onnx --model_path models/pretrained/RealESRGAN_x2plus.pth
"""

import argparse
import os
import sys
from typing import Any


def load_model(model_path: str) -> Any:
    """Load the RealESRGAN_x2plus model.

    Args:
        model_path: Path to .pth weights file.

    Returns:
        PyTorch module in eval mode.
    """
    import spandrel

    descriptor = spandrel.ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise TypeError(f"Expected an image-to-image model, got {type(descriptor).__name__}")

    # torch.onnx.export expects torch.nn.Module. Spandrel returns a descriptor wrapper;
    # descriptor.model is the underlying loaded module.
    descriptor.eval()
    model = descriptor.model
    model.eval()
    return model


def export_onnx(
    model: Any,
    input_h: int,
    input_w: int,
    output_path: str,
) -> None:
    """Export model to ONNX with fixed input size.

    Args:
        model: Loaded RRDBNet model.
        input_h: Input height in pixels.
        input_w: Input width in pixels.
        output_path: Path to save .onnx file.
    """
    import onnx
    import torch

    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32)

    print(f"  Export: input {input_w}x{input_h} -> output {input_w*2}x{input_h*2}")
    print(f"  File: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        # Fixed dimensions — no dynamic_axes
        # This gives TensorRT maximum optimization freedom
    )

    # New PyTorch exporter stores weights in a separate .data file.
    # Merge everything into a single .onnx for convenience.
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print("  Merging weights into a single file...")
        model_proto = onnx.load(output_path, load_external_data=True)
        onnx.save(model_proto, output_path, convert_attribute=True)
        os.remove(data_file)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export RealESRGAN_x2plus to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="RealESRGAN_x2plus.pth",
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/onnx",
        help="Output directory for ONNX files",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: File {args.model_path} not found.")
        sys.exit(1)

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    log("Loading model...")
    model = load_model(args.model_path)
    log("  Model loaded: RRDBNet x2plus (64 feat, 23 blocks)")

    # Export for both resolutions
    configs = [
        (720, 1280, "realesrgan_x2plus_720p.onnx"),  # 720p -> 1440p
        (1080, 1920, "realesrgan_x2plus_1080p.onnx"),  # FHD -> 4K
    ]

    for input_h, input_w, filename in configs:
        output_path = os.path.join(args.output_dir, filename)
        log(f"\n--- Export {filename} ---")
        export_onnx(model, input_h, input_w, output_path)

    log("\n=== All ONNX models exported ===")
    log(f"Files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
