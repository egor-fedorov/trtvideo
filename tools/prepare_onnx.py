#!/usr/bin/env python3
"""Prepare static ONNX models from dynamic ones.

If input dimensions are dynamic, creates files with fixed 720p and 1080p sizes.
If already static, reports and does nothing.

Usage:
    prepare-onnx models/onnx/model.onnx
"""

import argparse
import os
import sys

import onnx
from onnx.tools import update_model_dims


TARGETS = [
    {"name": "720p", "h": 720, "w": 1280},
    {"name": "1080p", "h": 1080, "w": 1920},
]


def get_dims(model):
    """Return names and dimensions of input/output tensors.

    Args:
        model: Loaded ONNX model.

    Returns:
        Tuple (inp_name, inp_dims, out_name, out_dims).
    """
    inp = model.graph.input[0]
    out = model.graph.output[0]
    inp_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    return inp.name, inp_dims, out.name, out_dims


def is_dynamic(dims):
    """Check for dynamic (zero) axes.

    Args:
        dims: List of tensor dimensions.

    Returns:
        True if there are dynamic axes.
    """
    return any(d == 0 for d in dims)


def main():
    parser = argparse.ArgumentParser(description="Prepare static ONNX models for TensorRT")
    parser.add_argument("onnx_path", help="Path to source ONNX file")
    parser.add_argument("--output_dir", default="./models/onnx", help="Output directory")
    parser.add_argument("--scale", type=int, default=2, help="Model scale factor (default: 2)")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"ERROR: File not found: {args.onnx_path}")
        sys.exit(1)

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    model = onnx.load(args.onnx_path)
    inp_name, inp_dims, out_name, out_dims = get_dims(model)
    basename = os.path.splitext(os.path.basename(args.onnx_path))[0]

    log(f"File:   {args.onnx_path}")
    log(f"Input:  {inp_name} {inp_dims}")
    log(f"Output: {out_name} {out_dims}")

    if not is_dynamic(inp_dims):
        h, w = inp_dims[2], inp_dims[3]
        log(f"\nDimensions are already static ({w}x{h}). Ready for build_engine.py.")
        log(f"Copy the file to {args.output_dir}/ if not already there.")
        return

    log(f"\nDynamic axes detected. Creating static variants (scale={args.scale}x)...")
    os.makedirs(args.output_dir, exist_ok=True)

    for target in TARGETS:
        h_in, w_in = target["h"], target["w"]
        h_out, w_out = h_in * args.scale, w_in * args.scale
        out_path = os.path.join(args.output_dir, f"{basename}_{target['name']}.onnx")

        # Reload — update_model_dims modifies the object in-place
        m = onnx.load(args.onnx_path)
        m = update_model_dims.update_inputs_outputs_dims(
            m,
            {inp_name: [1, 3, h_in, w_in]},
            {out_name: [1, 3, h_out, w_out]},
        )
        onnx.save(m, out_path)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log(f"  {target['name']}: {w_in}x{h_in} -> {w_out}x{h_out}  [{size_mb:.1f} MB]  {out_path}")

    log(f"\nDone. Next step: build-engine {args.output_dir}/<model>.onnx")


if __name__ == "__main__":
    main()
