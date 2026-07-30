#!/usr/bin/env python3
"""Compile ONNX to a TensorRT engine.

Builds an optimized engine for the current GPU. Compilation takes 5-15 minutes.

Usage:
    build-engine models/onnx/model_720p.onnx -o models/engines/model_720p.engine
    build-engine models/onnx/model.onnx \
        --min-shape input:1x3x360x640 \
        --opt-shape input:1x3x720x1280 \
        --max-shape input:1x3x1080x1920
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

trt: Any = None
TRT_LOGGER: Any = None
ShapeArg = tuple[str, tuple[int, ...]]
ProfileShapes = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def _load_tensorrt() -> None:
    """Load TensorRT only after argparse handled lightweight flags such as --help."""
    global TRT_LOGGER, trt
    if trt is not None:
        return

    import tensorrt as trt_module

    trt = trt_module
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def parse_shape_arg(value: str) -> ShapeArg:
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


def shape_has_dynamic_dims(shape: Sequence[int]) -> bool:
    """Return True when a TensorRT tensor shape contains dynamic axes."""
    return any(dim < 0 for dim in shape)


def validate_profile_shapes(
    input_tensor: Any,
    min_shape: ShapeArg | None,
    opt_shape: ShapeArg | None,
    max_shape: ShapeArg | None,
) -> ProfileShapes | None:
    """Validate optional TensorRT optimization profile shapes."""
    if min_shape is None and opt_shape is None and max_shape is None:
        return None

    if min_shape is None or opt_shape is None or max_shape is None:
        print(
            "ERROR: Dynamic profile requires all three flags: --min-shape, --opt-shape, --max-shape"
        )
        sys.exit(1)

    expected_name = input_tensor.name
    parsed = (
        ("--min-shape", min_shape),
        ("--opt-shape", opt_shape),
        ("--max-shape", max_shape),
    )
    for label, item in parsed:
        name, dims = item
        if name != expected_name:
            print(
                f"ERROR: {label} uses tensor '{name}', but ONNX input tensor is '{expected_name}'."
            )
            sys.exit(1)
        if len(dims) != len(input_tensor.shape):
            print(
                f"ERROR: {label} rank is {len(dims)}, "
                f"but ONNX input rank is {len(input_tensor.shape)}."
            )
            sys.exit(1)

    min_dims, opt_dims, max_dims = min_shape[1], opt_shape[1], max_shape[1]
    profile_axes = zip(min_dims, opt_dims, max_dims, strict=False)
    for axis, (min_dim, opt_dim, max_dim) in enumerate(profile_axes):
        if not (min_dim <= opt_dim <= max_dim):
            print(
                f"ERROR: Profile axis {axis} must satisfy min <= opt <= max, "
                f"got {min_dim} <= {opt_dim} <= {max_dim}."
            )
            sys.exit(1)

    return min_dims, opt_dims, max_dims


def _onnx_tensor_elem_type(tensor: Any) -> int | None:
    tensor_type = getattr(getattr(tensor, "type", None), "tensor_type", None)
    if tensor_type is None:
        return None
    elem_type = getattr(tensor_type, "elem_type", 0)
    return elem_type or None


def _onnx_model_precision(model: Any) -> str:
    """Infer model compute precision from ONNX tensor and initializer dtypes."""
    import onnx

    has_fp16 = False
    has_fp32 = False
    graph = model.graph

    for initializer in graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            has_fp16 = True
        elif initializer.data_type == onnx.TensorProto.FLOAT:
            has_fp32 = True

    for tensor in (*graph.input, *graph.output, *graph.value_info):
        elem_type = _onnx_tensor_elem_type(tensor)
        if elem_type == onnx.TensorProto.FLOAT16:
            has_fp16 = True
        elif elem_type == onnx.TensorProto.FLOAT:
            has_fp32 = True

    if has_fp16:
        return "fp16"
    if has_fp32:
        return "fp32"
    return "unknown"


def infer_onnx_precision(onnx_path: str) -> str:
    """Infer source ONNX precision for engine manifest metadata."""
    import onnx

    return _onnx_model_precision(onnx.load(onnx_path, load_external_data=False))


def _trt_dtype_precision(dtype: Any) -> str:
    if dtype == trt.DataType.HALF:
        return "fp16"
    if dtype == trt.DataType.FLOAT:
        return "fp32"
    return str(dtype)


def _engine_io_precision(input_dtype: Any, output_dtype: Any) -> str:
    input_precision = _trt_dtype_precision(input_dtype)
    output_precision = _trt_dtype_precision(output_dtype)
    if input_precision == output_precision:
        return input_precision
    return "mixed"


def sha256_file(path: str) -> str:
    """Return SHA256 for a model/build artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_timing_cache(config: Any, timing_cache_path: str | None) -> None:
    """Attach an optional TensorRT timing cache to the builder config."""
    if timing_cache_path is None:
        return

    cache_data = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, "rb") as f:
            cache_data = f.read()
        print(f"  Timing cache: loaded {timing_cache_path} ({len(cache_data)} bytes)")
    else:
        print(f"  Timing cache: new cache will be written to {timing_cache_path}")

    cache = config.create_timing_cache(cache_data)
    config.set_timing_cache(cache, False)


def save_timing_cache(config: Any, timing_cache_path: str | None) -> None:
    """Persist TensorRT timing cache after a successful build."""
    if timing_cache_path is None:
        return

    cache = config.get_timing_cache()
    if cache is None:
        print("  WARNING: TensorRT did not return a timing cache.")
        return

    os.makedirs(os.path.dirname(timing_cache_path) or ".", exist_ok=True)
    serialized_cache = cache.serialize()
    with open(timing_cache_path, "wb") as f:
        f.write(bytes(serialized_cache))
    print(f"  Timing cache saved: {timing_cache_path}")


def profile_manifest(profile_shapes: ProfileShapes | None) -> dict[str, list[int]] | None:
    """Serialize TensorRT profile shapes for engine manifest."""
    if profile_shapes is None:
        return None

    profile_min, profile_opt, profile_max = profile_shapes
    return {
        "min": list(profile_min),
        "opt": list(profile_opt),
        "max": list(profile_max),
    }


def write_engine_manifest(
    *,
    manifest_path: str,
    onnx_path: str,
    engine_path: str,
    input_tensor: Any,
    output_tensor: Any,
    profile_shapes: ProfileShapes | None,
    precision: str,
    io_precision: str,
    timing_cache_path: str | None,
) -> dict[str, Any]:
    """Write a sidecar manifest with engine compatibility metadata."""
    manifest = {
        "schema_version": 1,
        "engine_path": engine_path,
        "engine_sha256": sha256_file(engine_path),
        "onnx_path": onnx_path,
        "model_sha256": sha256_file(onnx_path),
        "onnx_opset": None,
        "tensorrt_version": trt.__version__,
        "precision": precision,
        "io_precision": io_precision,
        "input": {
            "name": input_tensor.name,
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
        },
        "output": {
            "name": output_tensor.name,
            "shape": list(output_tensor.shape),
            "dtype": str(output_tensor.dtype),
        },
        "input_profile": profile_manifest(profile_shapes),
        "builder_flags": [],
        "builder_optimization_level": 5,
        "timing_cache": timing_cache_path,
        "preprocess_version": "uint8_to_float_0_1",
        "postprocess_version": "float_0_1_to_uint8",
        "cuda_version": None,
        "gpu_compute_capability": None,
    }

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  Manifest saved: {manifest_path}")
    return manifest


def build_engine(
    onnx_path: str,
    engine_path: str,
    min_shape: ShapeArg | None = None,
    opt_shape: ShapeArg | None = None,
    max_shape: ShapeArg | None = None,
    timing_cache_path: str | None = None,
    manifest_path: str | None = None,
) -> bool:
    """Compile an ONNX file to a TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path to save the .engine file.
        min_shape: Optional optimization profile min shape.
        opt_shape: Optional optimization profile opt shape.
        max_shape: Optional optimization profile max shape.
        timing_cache_path: Optional TensorRT timing cache path.
        manifest_path: Optional sidecar manifest path.

    Returns:
        True on success, False on error.
    """
    _load_tensorrt()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
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
    load_timing_cache(config, timing_cache_path)

    # 8 GB workspace for intermediate buffers during optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    # Optimization level 5 — maximum (more kernel variants, longer compilation)
    config.builder_optimization_level = 5
    print("  Optimization level: 5 (maximum)")

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    precision = infer_onnx_precision(onnx_path)
    io_precision = _engine_io_precision(input_tensor.dtype, output_tensor.dtype)

    # Log input/output shapes
    print(f"  Precision: {precision}")
    print(f"  I/O precision: {io_precision}")
    print(f"  Input:  {input_tensor.name} {input_tensor.shape} {input_tensor.dtype}")
    print(f"  Output: {output_tensor.name} {output_tensor.shape} {output_tensor.dtype}")

    profile_shapes = validate_profile_shapes(input_tensor, min_shape, opt_shape, max_shape)

    if shape_has_dynamic_dims(input_tensor.shape):
        if profile_shapes is None:
            print(
                "\nERROR: ONNX input shape is dynamic, "
                "but no TensorRT optimization profile was provided."
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
        profile_min, profile_opt, profile_max = profile_shapes
        profile.set_shape(input_tensor.name, profile_min, profile_opt, profile_max)
        config.add_optimization_profile(profile)
        print(f"  Profile min: {profile_min}")
        print(f"  Profile opt: {profile_opt}")
        print(f"  Profile max: {profile_max}")
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
    save_timing_cache(config, timing_cache_path)

    if manifest_path is not None:
        write_engine_manifest(
            manifest_path=manifest_path,
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            profile_shapes=profile_shapes,
            precision=precision,
            io_precision=io_precision,
            timing_cache_path=timing_cache_path,
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("onnx", help="Path to ONNX file")
    parser.add_argument(
        "-o", "--output", default=None, help="Path to .engine file (default: next to ONNX)"
    )
    parser.add_argument(
        "--min-shape",
        help="Optimization profile min shape, e.g. input:1x3x360x640",
    )
    parser.add_argument(
        "--opt-shape",
        help="Optimization profile opt shape, e.g. input:1x3x720x1280",
    )
    parser.add_argument(
        "--max-shape",
        help="Optimization profile max shape, e.g. input:1x3x1080x1920",
    )
    parser.add_argument(
        "--timing-cache",
        default=None,
        help="Path to TensorRT timing cache file",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to engine manifest JSON (default: <engine>.json)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write engine manifest JSON",
    )
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

    _load_tensorrt()

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
    manifest_path = None if args.no_manifest else (args.manifest or f"{args.output}.json")
    # Profile validation needs the parsed network input, so it happens inside build_engine.
    success = build_engine(
        args.onnx,
        args.output,
        parsed_min,
        parsed_opt,
        parsed_max,
        timing_cache_path=args.timing_cache,
        manifest_path=manifest_path,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
