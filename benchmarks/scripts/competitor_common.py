"""Shared workload and command-plan helpers for competitor benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


class CompetitorError(RuntimeError):
    """Raised when a competitor benchmark contract is invalid."""


CommandSpec = list[list[str]]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with a benchmark-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompetitorError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompetitorError(f"Expected a JSON object in {path}")
    return value


def find_variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Find a canonical clip variant by name."""
    variants = manifest.get("clip", {}).get("variants", [])
    for variant in variants:
        if isinstance(variant, dict) and variant.get("name") == name:
            return variant
    available = ", ".join(str(item.get("name")) for item in variants)
    raise CompetitorError(f"Unknown variant {name!r}; available: {available}")


def find_model_variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Find a canonical model variant by name."""
    variants = manifest.get("model", {}).get("variants", [])
    for variant in variants:
        if isinstance(variant, dict) and variant.get("name") == name:
            return variant
    raise CompetitorError(f"Workload has no model variant {name!r}")


def competitor_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one pinned competitor definition."""
    if config.get("schema_version") != 1:
        raise CompetitorError("Unsupported competitor config schema_version")
    value = config.get("competitors", {}).get(name)
    if not isinstance(value, dict):
        raise CompetitorError(f"Competitor config has no {name!r} entry")
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
    """Build the common video output contract used by full-pipeline competitors."""
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


def command_spec(*commands: list[str]) -> CommandSpec:
    """Create an argv-only command or pipeline specification."""
    if not commands or any(not command for command in commands):
        raise CompetitorError("Command specification cannot be empty")
    return [list(command) for command in commands]


def display_command(spec: CommandSpec) -> str:
    """Render a command specification for a runbook without changing execution."""
    return " | ".join(shlex.join(command) for command in spec)


def asset_requirement(path: str, kind: str) -> dict[str, Any]:
    """Describe a required container path without requiring it during dry-run."""
    return {
        "kind": kind,
        "path": path,
        "present": Path(path).is_file(),
    }


def write_json_target(value: dict[str, Any], target: str | None) -> None:
    """Write JSON to a file or stdout; dry-run defaults to stdout."""
    if target is None or target == "-":
        json.dump(value, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def add_common_arguments(parser: argparse.ArgumentParser, *, engine: bool) -> None:
    """Add canonical workload, suite and dry-run arguments."""
    parser.add_argument("--manifest", required=True, help="Canonical workload manifest")
    parser.add_argument(
        "--competitors",
        default="/app/benchmarks/competitors.json",
        help="Pinned competitor definitions",
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
    parser.add_argument("--dry-run", action="store_true")


def plan_document(
    *,
    product: str,
    backend: str,
    comparison_class: str,
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
        "comparison_class": comparison_class,
        "workload_id": manifest.get("id"),
        "variant": variant_name,
        "implementation": implementation,
        "parameters": parameters,
        "commands": commands,
        "assets": assets,
        "limitations": limitations,
    }
