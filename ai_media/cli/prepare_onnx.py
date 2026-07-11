#!/usr/bin/env python3
"""Prepare static and optionally mixed-precision ONNX models.

If input dimensions are dynamic, creates files with fixed target sizes.
If already static, reports and does nothing unless FP16 conversion is requested.

Usage:
    prepare-onnx models/onnx/model.onnx
    prepare-onnx models/onnx/model.onnx --size 1280x720
    prepare-onnx models/onnx/model.onnx --size 1280x720 --precision fp16
"""

import argparse
import os
import sys
from typing import Any, Literal, TypedDict


class TargetSpec(TypedDict):
    name: str
    h: int
    w: int


TARGETS: list[TargetSpec] = [
    {"name": "720p", "h": 720, "w": 1280},
    {"name": "1080p", "h": 1080, "w": 1920},
]
ONNXPrecision = Literal["fp32", "fp16"]
KEEP_IO_TYPES_FOR_FP16 = True


def parse_size(size: str) -> TargetSpec:
    """Parse a WIDTHxHEIGHT target size."""
    normalized = size.lower().replace("*", "x")
    try:
        w_str, h_str = normalized.split("x", 1)
        w, h = int(w_str), int(h_str)
    except ValueError:
        print(f"ERROR: Invalid --size '{size}'. Expected WIDTHxHEIGHT, e.g. 1280x720.")
        sys.exit(1)

    if w <= 0 or h <= 0:
        print(f"ERROR: Invalid --size '{size}'. Width and height must be positive.")
        sys.exit(1)

    name = f"{w}x{h}"
    if (w, h) == (1280, 720):
        name = "720p"
    elif (w, h) == (1920, 1080):
        name = "1080p"

    return {"name": name, "h": h, "w": w}


def dim_to_value(dim: Any) -> int | str:
    """Return a readable ONNX dim value or symbolic name."""
    if dim.dim_param:
        return dim.dim_param
    return dim.dim_value


def get_dims(model: Any) -> tuple[str, list[int | str], str, list[int | str]]:
    """Return names and dimensions of input/output tensors.

    Args:
        model: Loaded ONNX model.

    Returns:
        Tuple (inp_name, inp_dims, out_name, out_dims).
    """
    inp = model.graph.input[0]
    out = model.graph.output[0]
    inp_dims = [dim_to_value(d) for d in inp.type.tensor_type.shape.dim]
    out_dims = [dim_to_value(d) for d in out.type.tensor_type.shape.dim]
    return inp.name, inp_dims, out.name, out_dims


def is_dynamic(dims: list[int | str]) -> bool:
    """Check for dynamic axes.

    Args:
        dims: List of tensor dimensions.

    Returns:
        True if there are dynamic axes.
    """
    for dim in dims:
        if isinstance(dim, str) and dim:
            return True
        if isinstance(dim, int) and dim <= 0:
            return True
    return False


def output_path_for_variant(
    output_dir: str,
    basename: str,
    variant_name: str | None,
    precision: ONNXPrecision,
) -> str:
    """Return output ONNX path for a static or converted variant."""
    suffixes: list[str] = []
    if variant_name:
        suffixes.append(variant_name)
    if precision == "fp16":
        suffixes.append("fp16")
    suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
    return os.path.join(output_dir, f"{basename}{suffix}.onnx")


def convert_to_mixed_precision(source_path: str, output_path: str) -> None:
    """Convert FP32 ONNX to mixed-precision FP16 using NVIDIA ModelOpt AutoCast."""
    try:
        import modelopt.onnx.autocast as autocast
        import onnx
    except ImportError as exc:
        print(
            "ERROR: FP16 ONNX conversion requires NVIDIA ModelOpt. "
            "Install the Docker/runtime dependencies or nvidia-modelopt[onnx]."
        )
        raise SystemExit(1) from exc

    converted_model = autocast.convert_to_mixed_precision(
        onnx_path=source_path,
        low_precision_type="fp16",
        keep_io_types=KEEP_IO_TYPES_FOR_FP16,
    )
    onnx.save(converted_model, output_path)


def save_static_variant(
    *,
    onnx_module: Any,
    update_model_dims_module: Any,
    source_path: str,
    output_path: str,
    inp_name: str,
    out_name: str,
    input_h: int,
    input_w: int,
    scale: int,
) -> None:
    """Save one static-shape ONNX variant."""
    h_out, w_out = input_h * scale, input_w * scale
    model = onnx_module.load(source_path)
    model = update_model_dims_module.update_inputs_outputs_dims(
        model,
        {inp_name: [1, 3, input_h, input_w]},
        {out_name: [1, 3, h_out, w_out]},
    )
    onnx_module.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare static ONNX models for TensorRT")
    parser.add_argument("onnx_path", help="Path to source ONNX file")
    parser.add_argument("--output_dir", default="./models/onnx", help="Output directory")
    parser.add_argument("--scale", type=int, default=2, help="Model scale factor (default: 2)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help=(
            "Output ONNX precision. fp16 uses NVIDIA ModelOpt AutoCast and keeps "
            "input/output tensors as FP32."
        ),
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
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"ERROR: File not found: {args.onnx_path}")
        sys.exit(1)

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    import onnx
    from onnx.tools import update_model_dims

    model = onnx.load(args.onnx_path)
    inp_name, inp_dims, out_name, out_dims = get_dims(model)
    basename = os.path.splitext(os.path.basename(args.onnx_path))[0]
    targets = [parse_size(size) for size in args.size] or TARGETS

    log(f"File:   {args.onnx_path}")
    log(f"Input:  {inp_name} {inp_dims}")
    log(f"Output: {out_name} {out_dims}")
    log(f"Precision: {args.precision}")

    if not is_dynamic(inp_dims):
        h, w = inp_dims[2], inp_dims[3]
        if args.precision == "fp32":
            log(f"\nDimensions are already static ({w}x{h}). Ready for build-engine.")
            log(f"Copy the file to {args.output_dir}/ if not already there.")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = output_path_for_variant(args.output_dir, basename, None, args.precision)
        convert_to_mixed_precision(args.onnx_path, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log(
            "\nStatic FP16 mixed-precision variant created "
            f"(FP32 I/O kept): [{size_mb:.1f} MB] {out_path}"
        )
        return

    log(f"\nDynamic axes detected. Creating static variants (scale={args.scale}x)...")
    os.makedirs(args.output_dir, exist_ok=True)

    for target in targets:
        h_in, w_in = target["h"], target["w"]
        h_out, w_out = h_in * args.scale, w_in * args.scale
        out_path = output_path_for_variant(
            args.output_dir,
            basename,
            str(target["name"]),
            args.precision,
        )
        static_path = out_path
        if args.precision == "fp16":
            static_path = os.path.join(
                args.output_dir,
                f".{basename}_{target['name']}.fp32.tmp.onnx",
            )

        # Reload: update_model_dims modifies the object in-place.
        save_static_variant(
            onnx_module=onnx,
            update_model_dims_module=update_model_dims,
            source_path=args.onnx_path,
            output_path=static_path,
            inp_name=inp_name,
            out_name=out_name,
            input_h=h_in,
            input_w=w_in,
            scale=args.scale,
        )

        if args.precision == "fp16":
            convert_to_mixed_precision(static_path, out_path)
            os.remove(static_path)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        io_note = " | FP32 I/O kept" if args.precision == "fp16" else ""
        log(
            f"  {target['name']}: {w_in}x{h_in} -> {w_out}x{h_out} "
            f"| {args.precision}{io_note}  [{size_mb:.1f} MB]  {out_path}"
        )

    log(f"\nDone. Next step: build-engine {args.output_dir}/<model>.onnx")


if __name__ == "__main__":
    main()
