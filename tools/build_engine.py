#!/usr/bin/env python3
"""Compile ONNX to TensorRT engine with FP16.

Builds an optimized engine for the current GPU. Compilation takes 5-15 minutes.

Usage:
    build-engine models/onnx/model_720p.onnx -o models/engines/model_720p.engine
    build-engine models/onnx/model.onnx --min-shape input:1x3x360x640 --opt-shape input:1x3x720x1280 --max-shape input:1x3x1080x1920
"""

import argparse
import os
import sys

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def parse_shape_arg(value: str) -> tuple[str, tuple[int, ...]]:
    """Parse NAME:DIMxDIMx... TensorRT profile shape."""
    if ":" not in value:
        print(f"ERROR: Invalid shape '{value}'. Expected NAME:DIMxDIMx..., e.g. input:1x3x720x1280")
        sys.exit(1)

    name, dims_str = value.split(":", 1)
    if not name:
        print(f"ERROR: Invalid shape '{value}'. Tensor name is empty.")
        sys.exit(1)

    try:
        dims = tuple(int(part) for part in dims_str.lower().split("x"))
    except ValueError:
        print(f"ERROR: Invalid shape '{value}'. Dimensions must be integers.")
        sys.exit(1)

    if not dims or any(dim <= 0 for dim in dims):
        print(f"ERROR: Invalid shape '{value}'. Dimensions must be positive.")
        sys.exit(1)

    return name, dims


def shape_has_dynamic_dims(shape) -> bool:
    """Return True when a TensorRT tensor shape contains dynamic axes."""
    return any(dim < 0 for dim in shape)


def validate_profile_shapes(
    input_tensor,
    min_shape: tuple[str, tuple[int, ...]] | None,
    opt_shape: tuple[str, tuple[int, ...]] | None,
    max_shape: tuple[str, tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Validate optional TensorRT optimization profile shapes."""
    provided = [shape for shape in (min_shape, opt_shape, max_shape) if shape is not None]
    if not provided:
        return None

    if len(provided) != 3:
        print("ERROR: Dynamic profile requires all three flags: --min-shape, --opt-shape, --max-shape")
        sys.exit(1)

    expected_name = input_tensor.name
    parsed = [min_shape, opt_shape, max_shape]
    for label, item in zip(("--min-shape", "--opt-shape", "--max-shape"), parsed):
        name, dims = item
        if name != expected_name:
            print(f"ERROR: {label} uses tensor '{name}', but ONNX input tensor is '{expected_name}'.")
            sys.exit(1)
        if len(dims) != len(input_tensor.shape):
            print(
                f"ERROR: {label} rank is {len(dims)}, but ONNX input rank is {len(input_tensor.shape)}."
            )
            sys.exit(1)

    min_dims, opt_dims, max_dims = (item[1] for item in parsed)
    for axis, (min_dim, opt_dim, max_dim) in enumerate(zip(min_dims, opt_dims, max_dims)):
        if not (min_dim <= opt_dim <= max_dim):
            print(
                f"ERROR: Profile axis {axis} must satisfy min <= opt <= max, "
                f"got {min_dim} <= {opt_dim} <= {max_dim}."
            )
            sys.exit(1)

    return min_dims, opt_dims, max_dims


def build_engine(
    onnx_path: str,
    engine_path: str,
    min_shape: tuple[str, tuple[int, ...]] | None = None,
    opt_shape: tuple[str, tuple[int, ...]] | None = None,
    max_shape: tuple[str, tuple[int, ...]] | None = None,
):
    """Compile an ONNX file to a TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path to save the .engine file.
        min_shape: Optional optimization profile min shape.
        opt_shape: Optional optimization profile opt shape.
        max_shape: Optional optimization profile max shape.

    Returns:
        True on success, False on error.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error: {parser.get_error(i)}")
            return False

    # Builder config
    config = builder.create_builder_config()

    # 8 GB workspace for intermediate buffers during optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    # FP16 — main speedup source on Tensor Core GPUs
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16: enabled (Tensor Cores)")
    else:
        print("  FP16: not supported on this GPU")

    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Optimization level 5 — maximum (more kernel variants, longer compilation)
    config.builder_optimization_level = 5
    print("  Optimization level: 5 (maximum)")

    # Log input/output shapes
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    print(f"  Input:  {input_tensor.name} {input_tensor.shape} {input_tensor.dtype}")
    print(f"  Output: {output_tensor.name} {output_tensor.shape} {output_tensor.dtype}")

    profile_shapes = validate_profile_shapes(input_tensor, min_shape, opt_shape, max_shape)

    if shape_has_dynamic_dims(input_tensor.shape):
        if profile_shapes is None:
            print(
                "\nERROR: ONNX input shape is dynamic, but no TensorRT optimization profile was provided."
            )
            print("  Option 1: create a static ONNX first:")
            print(f"    prepare-onnx {onnx_path} --size 1280x720")
            print("  Option 2: build with an explicit profile:")
            print(
                "    build-engine "
                f"{onnx_path} "
                "--min-shape input:1x3x360x640 "
                "--opt-shape input:1x3x720x1280 "
                "--max-shape input:1x3x1080x1920"
            )
            return False

        profile = builder.create_optimization_profile()
        min_shape, opt_shape, max_shape = profile_shapes
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"  Profile min: {min_shape}")
        print(f"  Profile opt: {opt_shape}")
        print(f"  Profile max: {max_shape}")
    elif profile_shapes is not None:
        print("  WARNING: Static ONNX input shape detected; ignoring optimization profile.")

    # Compile
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  ERROR: Compilation failed!")
        return False

    # Save
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("onnx", help="Path to ONNX file")
    parser.add_argument(
        "-o", "--output", default=None, help="Path to .engine file (default: next to ONNX)"
    )
    parser.add_argument("--min-shape", help="Optimization profile min shape, e.g. input:1x3x360x640")
    parser.add_argument("--opt-shape", help="Optimization profile opt shape, e.g. input:1x3x720x1280")
    parser.add_argument("--max-shape", help="Optimization profile max shape, e.g. input:1x3x1080x1920")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: File not found: {args.onnx}")
        sys.exit(1)

    if args.output is None:
        args.output = os.path.splitext(args.onnx)[0] + ".engine"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.verbose:
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE
    elif args.quiet:
        TRT_LOGGER.min_severity = trt.Logger.ERROR

    if not args.quiet:
        print(f"TensorRT version: {trt.__version__}\n")
        print(f"=== {os.path.basename(args.onnx)} ===")

    parsed_min = parse_shape_arg(args.min_shape) if args.min_shape else None
    parsed_opt = parse_shape_arg(args.opt_shape) if args.opt_shape else None
    parsed_max = parse_shape_arg(args.max_shape) if args.max_shape else None

    # Profile validation needs the parsed network input, so it happens inside build_engine.
    success = build_engine(args.onnx, args.output, parsed_min, parsed_opt, parsed_max)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
