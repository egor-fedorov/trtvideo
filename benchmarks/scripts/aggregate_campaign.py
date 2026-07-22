#!/usr/bin/env python3
"""Aggregate a rotated multi-product benchmark campaign."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import relative_artifact_path, sha256_file, write_json
from ai_media.benchmarking.runner import compute_suite_statistics

IMPLEMENTATIONS = {
    "ai-media": "ai-media-enhancer",
    "vstrt": "vs-mlrt",
    "vsgan": "VSGAN-tensorrt-docker",
}
ROUND_ORDERS = {
    1: ["ai-media", "vstrt", "vsgan"],
    2: ["vstrt", "vsgan", "ai-media"],
    3: ["vsgan", "ai-media", "vstrt"],
    4: ["vsgan", "vstrt", "ai-media"],
    5: ["ai-media", "vsgan", "vstrt"],
}
PUBLICATION_GAPS = [
    "Average CPU utilization is not collected yet",
    "Startup, steady-state and finalize/mux timing scopes are not collected yet",
    "Model-space RGB/float parity is not verified yet",
    "Product-output PSNR/SSIM and visual crops are not generated yet",
]


class CampaignError(RuntimeError):
    """Raised when campaign artifacts cannot form one comparable result."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"Expected a JSON object in {path}")
    return value


def _manifest_path(campaign_dir: Path, implementation: str, round_index: int) -> Path:
    return (
        campaign_dir
        / implementation
        / f"round-{round_index:02d}"
        / "run-01"
        / "manifest.json"
    )


def _load_rounds(campaign_dir: Path) -> list[dict[str, dict[str, Any]]]:
    rounds: list[dict[str, dict[str, Any]]] = []
    missing_started = False
    for round_index in ROUND_ORDERS:
        paths = {
            name: _manifest_path(campaign_dir, name, round_index)
            for name in IMPLEMENTATIONS
        }
        present = {name: path.is_file() for name, path in paths.items()}
        if not any(present.values()):
            missing_started = True
            continue
        if missing_started:
            raise CampaignError("Campaign rounds are not contiguous")
        missing = [name for name, exists in present.items() if not exists]
        if missing:
            raise CampaignError(
                f"Round {round_index} is incomplete; missing: {', '.join(missing)}"
            )
        rounds.append({name: _load_json(path) for name, path in paths.items()})
    if len(rounds) not in {3, 5}:
        raise CampaignError(f"Campaign requires 3 or 5 complete rounds, got {len(rounds)}")
    return rounds


def _asset_sha(manifest: dict[str, Any], name: str) -> str:
    value = manifest.get("assets", {}).get(name, {}).get("sha256")
    if not isinstance(value, str) or not value:
        raise CampaignError(f"Manifest has no SHA256 for {name}")
    return value


def _metric(manifest: dict[str, Any], *keys: str) -> float:
    value: Any = manifest.get("measured", {}).get("metrics", {})
    for key in keys:
        if not isinstance(value, dict):
            break
        value = value.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CampaignError(f"Manifest has no numeric measured metric: {'.'.join(keys)}")
    return float(value)


def _output_value(manifest: dict[str, Any], *keys: str) -> float:
    value: Any = manifest.get("measured", {})
    for key in keys:
        if not isinstance(value, dict):
            break
        value = value.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CampaignError(f"Manifest has no numeric output value: {'.'.join(keys)}")
    return float(value)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    implementation: str,
    round_index: int,
) -> None:
    expected_product = IMPLEMENTATIONS[implementation]
    if manifest.get("status") != "valid":
        raise CampaignError(f"{implementation} round {round_index} is not valid")
    if manifest.get("product") != expected_product:
        raise CampaignError(
            f"{implementation} round {round_index} has unexpected product"
        )
    if manifest.get("run_index") != 1:
        raise CampaignError(f"{implementation} round {round_index} is not a single run")
    reproducibility = manifest.get("reproducibility", {})
    if reproducibility.get("publishable") is not True:
        raise CampaignError(
            f"{implementation} round {round_index} is not reproducible"
        )


def _validate_common_contract(
    rounds: list[dict[str, dict[str, Any]]],
    *,
    root: Path,
    idle_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = rounds[0]["ai-media"]
    workload_id = first.get("workload_id")
    variant = first.get("variant")
    revision = first.get("environment", {}).get("image", {}).get("repository_revision")
    encoder = first.get("parameters", {}).get("encoder")
    input_sha = _asset_sha(first, "input")
    onnx_sha = _asset_sha(first, "onnx")
    gpu = first.get("environment", {}).get("gpu")
    cpu = first.get("environment", {}).get("cpu")
    frames = first.get("parameters", {}).get("frames")
    warmup_frames = first.get("parameters", {}).get("warmup_frames")
    if not isinstance(encoder, dict):
        raise CampaignError("Campaign manifests have no exact encoder contract")
    expected_revision = os.environ.get("AI_MEDIA_BUILD_REVISION")
    if expected_revision and expected_revision != "unknown" and revision != expected_revision:
        raise CampaignError(
            "Campaign repository revision does not match the aggregator image"
        )

    image_ids: dict[str, str] = {}
    engine_hashes: dict[str, str] = {}
    for round_index, round_data in enumerate(rounds, start=1):
        for implementation, manifest in round_data.items():
            _validate_manifest(
                manifest,
                implementation=implementation,
                round_index=round_index,
            )
            environment = manifest.get("environment", {})
            image = environment.get("image", {})
            checks = {
                "workload": (manifest.get("workload_id"), workload_id),
                "variant": (manifest.get("variant"), variant),
                "repository revision": (image.get("repository_revision"), revision),
                "source state": (str(image.get("source_dirty")), "0"),
                "GPU": (environment.get("gpu"), gpu),
                "CPU": (environment.get("cpu"), cpu),
                "frame count": (manifest.get("parameters", {}).get("frames"), frames),
                "warmup frame count": (
                    manifest.get("parameters", {}).get("warmup_frames"),
                    warmup_frames,
                ),
                "encoder contract": (
                    manifest.get("parameters", {}).get("encoder"),
                    encoder,
                ),
                "input SHA256": (_asset_sha(manifest, "input"), input_sha),
                "ONNX SHA256": (_asset_sha(manifest, "onnx"), onnx_sha),
            }
            for label, (actual, expected) in checks.items():
                if actual != expected:
                    raise CampaignError(
                        f"{implementation} round {round_index} changed {label}"
                    )
            image_id = image.get("id")
            if not isinstance(image_id, str) or not image_id:
                raise CampaignError(f"{implementation} has no image ID")
            previous_image = image_ids.setdefault(implementation, image_id)
            if image_id != previous_image:
                raise CampaignError(f"{implementation} image changed between rounds")
            engine_hash = _asset_sha(manifest, "engine")
            previous_engine = engine_hashes.setdefault(implementation, engine_hash)
            if engine_hash != previous_engine:
                raise CampaignError(f"{implementation} engine changed between rounds")

    if engine_hashes["ai-media"] != engine_hashes["vstrt"]:
        raise CampaignError("ai-media and vstrt must use the same serialized engine")

    workload_asset = first.get("assets", {}).get("workload_manifest", {})
    workload_path = root / str(workload_asset.get("path", ""))
    if not workload_path.is_file():
        raise CampaignError(f"Canonical workload manifest not found: {workload_path}")
    if sha256_file(workload_path) != workload_asset.get("sha256"):
        raise CampaignError("Canonical workload manifest SHA256 changed")
    workload = _load_json(workload_path)
    benchmark = workload.get("benchmark", {})
    canonical_checks = {
        "measured frames": (frames, benchmark.get("measured_frames")),
        "warmup frames": (warmup_frames, benchmark.get("warmup_frames")),
        "idle seconds": (idle_seconds, benchmark.get("idle_seconds")),
        "initial rounds": (3, benchmark.get("initial_runs")),
        "extra rounds": (2, benchmark.get("extra_runs_on_spread")),
    }
    for label, (actual, expected) in canonical_checks.items():
        if actual != expected:
            raise CampaignError(f"Campaign {label} is not canonical ({actual!r} != {expected!r})")
    return workload, {
        "workload_id": workload_id,
        "variant": variant,
        "repository_revision": revision,
        "input_sha256": input_sha,
        "onnx_sha256": onnx_sha,
        "encoder": encoder,
        "gpu": gpu,
        "cpu": cpu,
        "frames": frames,
        "warmup_frames": warmup_frames,
        "image_ids": image_ids,
        "engine_hashes": engine_hashes,
    }


def _implementation_statistics(
    rounds: list[dict[str, dict[str, Any]]], implementation: str
) -> dict[str, Any]:
    manifests = [round_data[implementation] for round_data in rounds]
    fps = [_metric(manifest, "end_to_end_fps") for manifest in manifests]
    statistics_report = compute_suite_statistics(fps)
    statistics_report.update(
        {
            "median_wall_time_sec": _median(
                [_metric(manifest, "wall_time_sec") for manifest in manifests]
            ),
            "median_gpu_utilization_percent": _median(
                [
                    _metric(manifest, "nvml", "utilization", "average_gpu_percent")
                    for manifest in manifests
                ]
            ),
            "median_power_w": _median(
                [
                    _metric(manifest, "nvml", "power", "average_w")
                    for manifest in manifests
                ]
            ),
            "median_joules_per_frame": _median(
                [
                    _metric(manifest, "nvml", "power", "joules_per_frame")
                    for manifest in manifests
                ]
            ),
            "median_peak_vram_mib": _median(
                [
                    _metric(manifest, "nvml", "memory", "peak_delta_mib")
                    for manifest in manifests
                ]
            ),
            "median_output_bitrate_mbps": _median(
                [
                    _output_value(
                        manifest,
                        "validation",
                        "observed",
                        "video_bitrate_bps",
                    )
                    / 1_000_000
                    for manifest in manifests
                ]
            ),
            "median_output_size_mib": _median(
                [
                    _output_value(manifest, "output", "size_bytes") / (1024 * 1024)
                    for manifest in manifests
                ]
            ),
        }
    )
    return statistics_report


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Rotated Benchmark Campaign",
        "",
        f"Status: `{summary['status']}`. Publication ready: `no`.",
        "",
        "| Implementation | Runs | Median FPS | vs ai-media | Median wall, s | "
        "GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps | Size, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        lines.append(
            f"| {result['product']} | {summary['parameters']['rounds']} | "
            f"{stats['median_fps']:.3f} | {result['relative_to_ai_media_percent']:+.2f}% | "
            f"{stats['median_wall_time_sec']:.2f} | "
            f"{stats['median_gpu_utilization_percent']:.2f} | "
            f"{stats['median_power_w']:.2f} | "
            f"{stats['median_joules_per_frame']:.3f} | "
            f"{stats['median_peak_vram_mib']:.1f} | "
            f"{stats['median_output_bitrate_mbps']:.3f} | "
            f"{stats['median_output_size_mib']:.1f} |"
        )
    lines.extend(["", "Publication gaps:"])
    lines.extend(f"- {gap}" for gap in summary["publication"]["errors"])
    lines.append("")
    return "\n".join(lines)


def aggregate_campaign(
    campaign_dir: Path,
    *,
    root: Path,
    idle_seconds: float,
) -> dict[str, Any]:
    """Validate and aggregate all completed rounds in one campaign directory."""
    rounds = _load_rounds(campaign_dir)
    workload, contract = _validate_common_contract(
        rounds,
        root=root,
        idle_seconds=idle_seconds,
    )
    implementation_results: dict[str, Any] = {}
    for name, product in IMPLEMENTATIONS.items():
        implementation_results[name] = {
            "product": product,
            "image_id": contract["image_ids"][name],
            "engine_sha256": contract["engine_hashes"][name],
            "statistics": _implementation_statistics(rounds, name),
        }

    ai_fps = implementation_results["ai-media"]["statistics"]["median_fps"]
    for result in implementation_results.values():
        median_fps = result["statistics"]["median_fps"]
        result["relative_to_ai_media_percent"] = (median_fps / ai_fps - 1) * 100

    spread_threshold = float(workload["benchmark"]["spread_threshold"])
    unstable = [
        name
        for name, result in implementation_results.items()
        if result["statistics"]["relative_spread"] > spread_threshold
    ]
    needs_extra = len(rounds) == 3 and bool(unstable)
    status = "needs-extra-runs" if needs_extra else ("unstable" if unstable else "valid")
    summary = {
        "schema_version": 1,
        "document_type": "benchmark-campaign",
        "status": status,
        "scope": "rotated-campaign",
        "publishable": False,
        "publication": {
            "ready": False,
            "errors": PUBLICATION_GAPS,
        },
        "workload_id": contract["workload_id"],
        "variant": contract["variant"],
        "parameters": {
            "rounds": len(rounds),
            "warmup_frames": contract["warmup_frames"],
            "measured_frames": contract["frames"],
            "idle_seconds": idle_seconds,
            "spread_threshold": spread_threshold,
            "encoder": contract["encoder"],
        },
        "environment": {
            "repository_revision": contract["repository_revision"],
            "gpu": contract["gpu"],
            "cpu": contract["cpu"],
        },
        "assets": {
            "input_sha256": contract["input_sha256"],
            "onnx_sha256": contract["onnx_sha256"],
        },
        "rounds": [
            {
                "index": index,
                "order": ROUND_ORDERS[index],
                "manifests": {
                    name: relative_artifact_path(
                        _manifest_path(campaign_dir, name, index), root
                    )
                    for name in IMPLEMENTATIONS
                },
            }
            for index in range(1, len(rounds) + 1)
        ],
        "unstable_implementations": unstable,
        "needs_extra_runs": needs_extra,
        "implementations": implementation_results,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate a rotated benchmark campaign")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--root", default="/app")
    parser.add_argument("--idle-seconds", type=float, required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--request-extra-exit-code",
        action="store_true",
        help="Exit with code 3 when two extra rotated rounds are required",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign_dir = Path(args.campaign_dir)
    json_path = Path(args.json) if args.json else campaign_dir / "campaign.json"
    markdown_path = (
        Path(args.markdown) if args.markdown else campaign_dir / "results.md"
    )
    try:
        summary = aggregate_campaign(
            campaign_dir,
            root=Path(args.root),
            idle_seconds=args.idle_seconds,
        )
        write_json(json_path, summary)
        markdown_path.write_text(_markdown(summary), encoding="utf-8")
    except (CampaignError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(
        f"Campaign {summary['status']}: {len(summary['rounds'])} rotated rounds; "
        f"results: {json_path}",
        file=sys.stderr,
    )
    if summary["needs_extra_runs"] and args.request_extra_exit_code:
        sys.exit(3)
    if summary["status"] == "unstable":
        sys.exit(2)


if __name__ == "__main__":
    main()
