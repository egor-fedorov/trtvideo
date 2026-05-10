#!/usr/bin/env python3
"""Compile ONNX to TensorRT engine with FP16.

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

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ShapeArg = tuple[str, tuple[int, ...]]
ProfileShapes = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def parse_shape_arg(value: str) -> ShapeArg:
    """Parse NAME:DIMxDIMx... TensorRT profile shape."""
    if ":" not in value:
        print(
            f"ERROR: Invalid shape '{value}'. "
            "Expected NAME:DIMxDIMx..., e.g. input:1x3x720x1280"
        )
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
            "ERROR: Dynamic profile requires all three flags: "
            "--min-shape, --opt-shape, --max-shape"
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
                f"ERROR: {label} uses tensor '{name}', "
                f"but ONNX input tensor is '{expected_name}'."
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


def _registry_manifest_path(path: str) -> str:
    """Return a registry manifest file path from a directory or JSON path."""
    if path.endswith(".json"):
        return path
    return os.path.join(path, "manifest.json")


def _relative_to(path: str, base_dir: str) -> str:
    """Store registry paths relative to the registry file when practical."""
    return os.path.relpath(os.path.abspath(path), os.path.abspath(base_dir))


def update_model_registry(
    *,
    registry_path: str,
    engine_manifest: dict[str, Any],
    manifest_path: str,
) -> None:
    """Upsert one engine entry into a model registry manifest."""
    registry_path = _registry_manifest_path(registry_path)
    registry_dir = os.path.dirname(registry_path) or "."
    os.makedirs(registry_dir, exist_ok=True)

    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        if not isinstance(registry, dict):
            print(f"ERROR: Registry manifest must be a JSON object: {registry_path}")
            sys.exit(1)
    else:
        registry = {"schema_version": 1, "engines": []}

    engines = registry.setdefault("engines", [])
    if not isinstance(engines, list):
        print(f"ERROR: Registry 'engines' must be a list: {registry_path}")
        sys.exit(1)

    entry = dict(engine_manifest)
    entry["engine_path"] = _relative_to(engine_manifest["engine_path"], registry_dir)
    entry["manifest_path"] = _relative_to(manifest_path, registry_dir)

    upserted = False
    for idx, existing in enumerate(engines):
        if not isinstance(existing, dict):
            continue
        same_engine = existing.get("engine_path") == entry["engine_path"]
        same_manifest = existing.get("manifest_path") == entry["manifest_path"]
        if same_engine or same_manifest:
            engines[idx] = entry
            upserted = True
            break

    if not upserted:
        engines.append(entry)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  Registry updated: {registry_path}")


def write_engine_manifest(
    *,
    manifest_path: str,
    onnx_path: str,
    engine_path: str,
    input_tensor: Any,
    output_tensor: Any,
    profile_shapes: ProfileShapes | None,
    fp16_enabled: bool,
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
        "precision": "fp16" if fp16_enabled else "fp32",
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
        "builder_flags": [
            flag
            for flag, enabled in (
                ("FP16", fp16_enabled),
                ("PREFER_PRECISION_CONSTRAINTS", True),
            )
            if enabled
        ],
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
    fp16: bool = True,
    timing_cache_path: str | None = None,
    manifest_path: str | None = None,
    registry_path: str | None = None,
) -> bool:
    """Compile an ONNX file to a TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path to save the .engine file.
        min_shape: Optional optimization profile min shape.
        opt_shape: Optional optimization profile opt shape.
        max_shape: Optional optimization profile max shape.
        fp16: Enable FP16 kernels when hardware supports them.
        timing_cache_path: Optional TensorRT timing cache path.
        manifest_path: Optional sidecar manifest path.
        registry_path: Optional model registry manifest path to update.

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
    load_timing_cache(config, timing_cache_path)

    # 8 GB workspace for intermediate buffers during optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    # FP16 — main speedup source on Tensor Core GPUs
    fp16_enabled = False
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        fp16_enabled = True
        print("  FP16: enabled (Tensor Cores)")
    elif fp16:
        print("  FP16: not supported on this GPU")
    else:
        print("  FP16: disabled")

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
        engine_manifest = write_engine_manifest(
            manifest_path=manifest_path,
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            profile_shapes=profile_shapes,
            fp16_enabled=fp16_enabled,
            timing_cache_path=timing_cache_path,
        )
        if registry_path is not None:
            update_model_registry(
                registry_path=registry_path,
                engine_manifest=engine_manifest,
                manifest_path=manifest_path,
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
        "--fp16",
        dest="fp16",
        action="store_true",
        default=True,
        help="Enable FP16 kernels when supported (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Build without FP16 kernels",
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
        "--registry",
        default=None,
        help="Update model registry manifest JSON or model directory",
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
    if args.registry and manifest_path is None:
        print("ERROR: --registry requires engine manifest output; remove --no-manifest.")
        sys.exit(1)

    # Profile validation needs the parsed network input, so it happens inside build_engine.
    success = build_engine(
        args.onnx,
        args.output,
        parsed_min,
        parsed_opt,
        parsed_max,
        fp16=args.fp16,
        timing_cache_path=args.timing_cache,
        manifest_path=manifest_path,
        registry_path=args.registry,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
