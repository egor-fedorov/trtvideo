#!/usr/bin/env python3
"""Export a Spandrel-supported x2 image model to static ONNX variants.

Creates ONNX files for selected resolutions, defaulting to 720p and 1080p.

Usage:
    export-onnx --model_path models/pretrained/model.pth --name model
    export-onnx --model_path models/pretrained/model.pth --size 1280x720
"""

import argparse
import os
import sys
from typing import Any

from trtvideo.cli.prepare_onnx import TARGETS, TargetSpec, parse_size


def load_model(model_path: str) -> Any:
    """Load an image-to-image model supported by Spandrel.

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


def freeze_reparameterized_convs(model: Any) -> int:
    """Replace mutable Spandrel Conv3XC blocks with their fused eval convolutions."""
    import torch
    from torch import nn

    replaced = 0

    def freeze_children(module: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            update_params = getattr(child, "update_params", None)
            eval_conv = getattr(child, "eval_conv", None)
            if (
                type(child).__name__ != "Conv3XC"
                or not callable(update_params)
                or not isinstance(eval_conv, nn.Module)
            ):
                freeze_children(child)
                continue

            # Conv3XC normally rewrites fused weights in every eval forward. Capture
            # that equivalent graph once so torch.export sees no module mutations.
            with torch.no_grad():
                update_params()
            layers: list[nn.Module] = [eval_conv]
            if bool(getattr(child, "has_relu", False)):
                layers.append(nn.LeakyReLU(negative_slope=0.05))
            setattr(module, name, nn.Sequential(*layers))
            replaced += 1

    freeze_children(model)
    return replaced


def export_onnx(
    model: Any,
    input_h: int,
    input_w: int,
    output_path: str,
) -> None:
    """Export model to ONNX with fixed input size.

    Args:
        model: Loaded x2 image model.
        input_h: Input height in pixels.
        input_w: Input width in pixels.
        output_path: Path to save .onnx file.
    """
    import onnx
    import torch

    replaced = freeze_reparameterized_convs(model)
    if replaced:
        print(f"  Reparameterized {replaced} mutable convolution blocks")
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32)

    print(f"  Export: input {input_w}x{input_h} -> output {input_w*2}x{input_h*2}")
    print(f"  File: {output_path}")

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
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


def export_filename(model_name: str, height: int) -> str:
    """Return a deterministic static ONNX filename."""
    return f"{model_name}_{height}p.onnx"


def export_filename_for_target(model_name: str, target: TargetSpec) -> str:
    """Return a deterministic filename for a parsed target resolution."""
    return f"{model_name}_{target['name']}.onnx"


def build_parser() -> argparse.ArgumentParser:
    """Create the exporter CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export a Spandrel-supported x2 image model to static ONNX"
    )
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
    parser.add_argument(
        "--name",
        default="realesrgan_x2plus",
        help="Output model basename (default: realesrgan_x2plus)",
    )
    parser.add_argument(
        "--size",
        action="append",
        default=[],
        help="Target input size WIDTHxHEIGHT. Can be repeated. Default: 1280x720 and 1920x1080.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
    log(f"  Model loaded: {type(model).__name__}")

    targets = [parse_size(size) for size in args.size] or TARGETS

    for target in targets:
        input_h, input_w = target["h"], target["w"]
        filename = export_filename_for_target(args.name, target)
        output_path = os.path.join(args.output_dir, filename)
        log(f"\n--- Export {filename} ---")
        export_onnx(model, input_h, input_w, output_path)

    log("\n=== All ONNX models exported ===")
    log(f"Files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
