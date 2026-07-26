#!/usr/bin/env python3
"""Execute the canonical ai-media-enhancer benchmark workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the canonical workload manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read workload manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def find_variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one clip variant or fail with available names."""
    variants = manifest.get("clip", {}).get("variants", [])
    for variant in variants:
        if variant.get("name") == name:
            return variant
    available = ", ".join(str(variant.get("name")) for variant in variants)
    raise ValueError(f"Unknown variant {name!r}; available: {available}")


def build_command(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    """Build the installed benchmark-upscale invocation from pinned workload values."""
    benchmark = manifest["benchmark"]
    variant = find_variant(manifest, args.variant)
    output = variant["benchmark_output"]
    command = [
        "benchmark-upscale",
        "--engine",
        args.engine,
        "--input",
        variant["path"],
        "--backend",
        "nvcodec",
        "--gpu-id",
        str(args.gpu_id),
        "--bitrate-mbps",
        str(output["bitrate_mbps"]),
        "--warmup-frames",
        str(
            args.warmup_frames
            if args.warmup_frames is not None
            else benchmark["warmup_frames"]
        ),
        "--frames",
        str(args.frames if args.frames is not None else benchmark["measured_frames"]),
        "--runs",
        str(args.runs if args.runs is not None else benchmark["initial_runs"]),
        "--extra-runs",
        str(
            args.extra_runs
            if args.extra_runs is not None
            else benchmark["extra_runs_on_spread"]
        ),
        "--spread-threshold",
        str(benchmark["spread_threshold"]),
        "--idle-seconds",
        str(args.idle_seconds if args.idle_seconds is not None else benchmark["idle_seconds"]),
        "--nvml-sample-ms",
        str(benchmark["nvml_sample_interval_ms"]),
        "--output-dir",
        args.output_dir,
        "--workload-manifest",
        args.manifest,
        "--variant",
        args.variant,
    ]
    if args.json is not None:
        command.extend(["--json", args.json])
    if args.cuda_graph:
        command.append("--cuda-graph")
    if args.keep_outputs:
        command.append("--keep-outputs")
    if args.skip_bitrate_validation:
        command.append("--skip-bitrate-validation")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical ai-media benchmark")
    parser.add_argument("--manifest", required=True, help="Canonical workload manifest")
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True, help="Engine built for the selected variant")
    parser.add_argument("--output-dir", required=True, help="Git-ignored raw artifact directory")
    parser.add_argument("--json", default=None, help="Optional summary JSON path or '-'")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--extra-runs", type=int, default=None)
    parser.add_argument("--idle-seconds", type=float, default=None)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--skip-bitrate-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(Path(args.manifest))
        command = build_command(args, manifest)
        result = subprocess.run(command, check=False)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
