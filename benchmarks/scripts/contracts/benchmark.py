"""Shared configuration and plan contracts for benchmark runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CompetitorError(RuntimeError):
    """Raised when an external benchmark contract is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with a benchmark-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompetitorError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompetitorError(f"Expected a JSON object in {path}")
    return value


def implementation_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one pinned benchmark implementation definition."""
    if config.get("schema_version") != 1:
        raise CompetitorError("Unsupported implementation config schema_version")
    value = config.get("implementations", {}).get(name)
    if not isinstance(value, dict):
        raise CompetitorError(f"Implementation config has no {name!r} entry")
    return value


def benchmark_value(
    override: int | float | None,
    benchmark: dict[str, Any],
    key: str,
) -> int | float:
    """Resolve a numeric CLI override against the canonical workload."""
    value = override if override is not None else benchmark.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise CompetitorError(f"Invalid benchmark value for {key}: {value!r}")
    return value


def benchmark_parameters(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve shared 3+2 suite parameters."""
    benchmark = manifest.get("benchmark", {})
    values = {
        "frames": int(benchmark_value(args.frames, benchmark, "measured_frames")),
        "warmup_frames": int(
            benchmark_value(args.warmup_frames, benchmark, "warmup_frames")
        ),
        "initial_runs": int(benchmark_value(args.runs, benchmark, "initial_runs")),
        "extra_runs_on_spread": int(
            benchmark_value(
                args.extra_runs,
                benchmark,
                "extra_runs_on_spread",
            )
        ),
        "idle_seconds": float(
            benchmark_value(args.idle_seconds, benchmark, "idle_seconds")
        ),
        "spread_threshold": float(benchmark.get("spread_threshold", 0.05)),
        "nvml_sample_interval_ms": int(
            benchmark.get("nvml_sample_interval_ms", 100)
        ),
    }
    for key in ("frames", "warmup_frames", "initial_runs"):
        if values[key] <= 0:
            raise CompetitorError(f"{key} must be greater than zero")
    if values["extra_runs_on_spread"] < 0 or values["idle_seconds"] < 0:
        raise CompetitorError("extra runs and idle time cannot be negative")
    if not 0 <= values["spread_threshold"] < 1:
        raise CompetitorError("spread threshold must be in the range [0, 1)")
    return values


def output_contract(
    manifest: dict[str, Any],
    variant: dict[str, Any],
    *,
    frames: int,
    enforce_bitrate: bool,
) -> dict[str, Any]:
    """Build the common video output contract for external implementations."""
    output = variant["benchmark_output"]
    return {
        "width": output["width"],
        "height": output["height"],
        "fps": manifest["clip"]["fps"],
        "frames": frames,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "has_b_frames": 0,
        "gop_frames": _fps_numerator(manifest["clip"]["fps"]),
        "target_bitrate_mbps": output["bitrate_mbps"] if enforce_bitrate else None,
        "bitrate_tolerance": 0.10,
        "require_monotonic_pts": True,
        "require_monotonic_dts": True,
    }


def _fps_numerator(value: str) -> int:
    numerator, denominator = (int(part) for part in value.split("/", 1))
    return max(1, round(numerator / denominator))


def asset_requirement(path: str, kind: str) -> dict[str, Any]:
    """Describe a required container path without requiring it during dry-run."""
    return {
        "kind": kind,
        "path": path,
        "present": Path(path).is_file(),
    }


def add_common_arguments(parser: argparse.ArgumentParser, *, engine: bool) -> None:
    """Add canonical workload, suite and dry-run arguments."""
    parser.add_argument("--manifest", required=True, help="Canonical workload manifest")
    parser.add_argument(
        "--implementations",
        default="/app/benchmarks/implementations.json",
        help="Pinned benchmark implementation definitions",
    )
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    if engine:
        parser.add_argument("--engine", required=True, help="TensorRT engine for the variant")
    parser.add_argument("--output-dir", required=True, help="Raw benchmark artifact directory")
    parser.add_argument("--json", default=None, help="Plan/summary JSON path or '-'")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--extra-runs", type=int, default=None)
    parser.add_argument("--idle-seconds", type=float, default=None)
    parser.add_argument(
        "--skip-bitrate-validation",
        action="store_true",
        help="Record but do not enforce average bitrate (short smoke runs only)",
    )
    parser.add_argument("--dry-run", action="store_true")


def plan_document(
    *,
    product: str,
    backend: str,
    implementation: dict[str, Any],
    manifest: dict[str, Any],
    variant_name: str,
    parameters: dict[str, Any],
    commands: dict[str, Any],
    assets: list[dict[str, Any]],
    limitations: list[str],
) -> dict[str, Any]:
    """Build the common machine-readable dry-run schema."""
    return {
        "schema_version": 1,
        "document_type": "benchmark-plan",
        "product": product,
        "backend": backend,
        "workload_id": manifest.get("id"),
        "benchmark_contract_version": manifest.get("benchmark", {}).get(
            "contract_version"
        ),
        "variant": variant_name,
        "implementation": implementation,
        "parameters": parameters,
        "commands": commands,
        "assets": assets,
        "limitations": limitations,
    }
