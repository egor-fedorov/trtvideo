#!/usr/bin/env python3
"""Run reproducible process-level video upscale benchmark suites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.scripts.contracts.engine import EngineContractError
from benchmarks.scripts.runners.trtvideo_suite import (
    BenchmarkConfig,
    BenchmarkError,
    run_suite,
)
from benchmarks.scripts.runtime.io import write_summary_target
from benchmarks.scripts.runtime.nvml import NvmlError


def build_parser() -> argparse.ArgumentParser:
    """Build the end-to-end benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark-upscale",
        description=(
            "Benchmark unprofiled upscale subprocesses with external wall time, "
            "NVML sampling and FFmpeg output validation"
        ),
    )
    parser.add_argument("--engine", required=True, help="Static TensorRT engine path")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run manifests, logs and raw NVML samples",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional additional suite JSON path, or '-' for stdout",
    )
    parser.add_argument("--frames", type=int, default=1000, help="Measured frames per run")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=100,
        help="Frames processed by a separate discarded process before each measured run",
    )
    parser.add_argument("--runs", type=int, default=3, help="Initial measured run count")
    parser.add_argument(
        "--extra-runs",
        type=int,
        default=2,
        help="Runs added when relative FPS spread exceeds the threshold",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        default=0.05,
        help="Maximum stable relative FPS spread (default: 0.05)",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=10.0,
        help="Fixed idle interval between measured runs",
    )
    parser.add_argument(
        "--nvml-sample-ms",
        type=int,
        default=100,
        help="External NVML sample interval in milliseconds",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="Visible GPU index")
    parser.add_argument(
        "--bitrate-mbps",
        type=float,
        default=None,
        help="Required explicit H.264 NVENC bitrate",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep valid warmup and measured MP4 files",
    )
    parser.add_argument(
        "--skip-bitrate-validation",
        action="store_true",
        help=(
            "Record but do not enforce average bitrate (short smoke or tuning reconnaissance only)"
        ),
    )
    parser.add_argument(
        "--workload-manifest",
        default=None,
        help="Optional canonical workload manifest for asset verification",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Canonical workload variant used with --workload-manifest",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert CLI values to the immutable runner contract."""
    return BenchmarkConfig(
        engine=Path(args.engine),
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        gpu_id=args.gpu_id,
        frames=args.frames,
        warmup_frames=args.warmup_frames,
        initial_runs=args.runs,
        extra_runs=args.extra_runs,
        spread_threshold=args.spread_threshold,
        idle_seconds=args.idle_seconds,
        sample_interval_ms=args.nvml_sample_ms,
        bitrate_mbps=args.bitrate_mbps,
        keep_outputs=args.keep_outputs,
        validate_bitrate=not args.skip_bitrate_validation,
        workload_manifest=(
            Path(args.workload_manifest) if args.workload_manifest is not None else None
        ),
        variant=args.variant,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary, returncode = run_suite(config_from_args(args))
        write_summary_target(args.json, summary)
    except (BenchmarkError, EngineContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except NvmlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("ERROR: benchmark interrupted", file=sys.stderr)
        sys.exit(130)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
