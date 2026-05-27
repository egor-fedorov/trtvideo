#!/usr/bin/env python3
"""Benchmark video upscale backends and write machine-readable JSON."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable

SUPPORTED_BACKENDS = ("ffmpeg", "nvcodec")


def parse_backends(value: str) -> list[str]:
    """Parse comma-separated backend list."""
    backends = [item.strip() for item in value.split(",") if item.strip()]
    if not backends:
        print("ERROR: --backend must contain at least one backend", file=sys.stderr)
        sys.exit(1)

    invalid = [backend for backend in backends if backend not in SUPPORTED_BACKENDS]
    if invalid:
        print(f"ERROR: Unsupported backend(s): {', '.join(invalid)}", file=sys.stderr)
        print(f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}", file=sys.stderr)
        sys.exit(1)

    return backends


def print_stderr_block(value: str) -> None:
    """Print captured process output to stderr without adding extra blank lines."""
    print(value, end="" if value.endswith("\n") else "\n", file=sys.stderr)


def build_backend_command(
    *,
    backend: str,
    args: argparse.Namespace,
    output_path: str,
    profile_json_path: str,
) -> list[str]:
    """Build one backend command invocation."""
    command = [
        "upscale",
        "--backend",
        backend,
        "--input",
        args.input,
        "--output",
        output_path,
        "--gpu-id",
        str(args.gpu_id),
        "--warmup-frames",
        str(args.warmup_frames),
        "--profile-json",
        profile_json_path,
        "--quiet",
    ]
    if args.cuda_graph:
        command += ["--cuda-graph"]
    if args.bitrate_mbps is not None:
        command += ["--bitrate-mbps", str(args.bitrate_mbps)]

    if args.frames > 0:
        command += ["--max-frames", str(args.frames + args.warmup_frames)]

    if args.engine:
        command += ["--engine", args.engine]
    else:
        command += ["--model", args.model]
        if args.engine_precision:
            command += ["--engine-precision", args.engine_precision]
        if args.engine_io_precision:
            command += ["--engine-io-precision", args.engine_io_precision]

    return command


def run_command(command: list[str]) -> None:
    """Run a benchmark child process and fail with captured output on error."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        print(f"ERROR: failed to run benchmark command: {exc}", file=sys.stderr)
        print(f"Command: {' '.join(command)}", file=sys.stderr)
        sys.exit(1)

    if result.returncode == 0:
        return

    if result.stdout:
        print_stderr_block(result.stdout)
    if result.stderr:
        print_stderr_block(result.stderr)
    print(f"ERROR: benchmark command failed: {' '.join(command)}", file=sys.stderr)
    sys.exit(result.returncode)


def run_benchmarks(
    *,
    backends: Iterable[str],
    args: argparse.Namespace,
    output_dir: str,
) -> list[dict]:
    """Run selected backends and collect per-backend JSON profiles."""
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    for backend in backends:
        profile_json_path = os.path.join(output_dir, f"{backend}.profile.json")
        output_path = os.path.join(output_dir, f"{backend}.mp4")
        command = build_backend_command(
            backend=backend,
            args=args,
            output_path=output_path,
            profile_json_path=profile_json_path,
        )
        if not args.quiet:
            print(f"Benchmarking {backend}: {' '.join(command)}", file=sys.stderr)
        run_command(command)

        with open(profile_json_path, encoding="utf-8") as f:
            result = json.load(f)
        result["command"] = command
        results.append(result)

    return results


def write_report(path: str, report: dict) -> None:
    """Write final benchmark report."""
    if path == "-":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmark-upscale",
        description="Benchmark TensorRT video upscale backends",
    )
    engine_source = parser.add_mutually_exclusive_group(required=True)
    engine_source.add_argument("--engine", help="Path to .engine file")
    engine_source.add_argument("--model", help="Path to model registry directory or manifest JSON")
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument(
        "--backend",
        default="ffmpeg,nvcodec",
        help="Comma-separated backend list: ffmpeg,nvcodec",
    )
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warmup")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Warmup frames to skip")
    parser.add_argument(
        "--json",
        required=True,
        help="Output benchmark JSON path, or '-' for stdout",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for temporary backend outputs",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--quiet", action="store_true", help="Suppress benchmark progress output")
    parser.add_argument(
        "--bitrate-mbps",
        type=float,
        default=None,
        help="NVENC target bitrate in Mbps; forwarded to nvcodec backend",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Experimental: benchmark TensorRT CUDA Graph capture",
    )
    parser.add_argument(
        "--engine-precision",
        choices=["fp16", "fp32"],
        default=None,
        help="Preferred precision when selecting an engine from --model",
    )
    parser.add_argument(
        "--engine-io-precision",
        choices=["fp16", "fp32"],
        default=None,
        help="Preferred input/output binding precision when selecting an engine from --model",
    )
    args = parser.parse_args()

    backends = parse_backends(args.backend)

    if args.output_dir:
        results = run_benchmarks(backends=backends, args=args, output_dir=args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="ai-media-benchmark-") as tmp_dir:
            results = run_benchmarks(backends=backends, args=args, output_dir=tmp_dir)

    report = {
        "input": args.input,
        "engine": args.engine,
        "model": args.model,
        "frames": args.frames,
        "warmup_frames": args.warmup_frames,
        "backends": backends,
        "results": results,
    }
    write_report(args.json, report)
    if not args.quiet:
        target = "stdout" if args.json == "-" else args.json
        print(f"Benchmark JSON written: {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
