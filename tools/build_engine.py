#!/usr/bin/env python3
"""Compile ONNX to TensorRT engine with FP16.

Builds an optimized engine for the current GPU. Compilation takes 5-15 minutes.

Usage:
    build-engine models/onnx/model_720p.onnx -o models/engines/model_720p.engine
"""

import argparse
import os
import sys

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, engine_path: str):
    """Compile an ONNX file to a TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path to save the .engine file.

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
    success = build_engine(args.onnx, args.output)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
