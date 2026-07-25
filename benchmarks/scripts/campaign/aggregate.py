#!/usr/bin/env python3
"""Aggregate a rotated multi-product benchmark campaign."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import relative_artifact_path, sha256_file, write_json
from ai_media.benchmarking.suite import compute_suite_statistics
from benchmarks.scripts.campaign.core import (
    CONFIG_NAME,
    EVENT_LOG_NAME,
    EXECUTION_PROFILES,
    IMPLEMENTATIONS,
    ROUND_ORDERS,
    CampaignEventError,
    load_campaign_config,
    load_events,
    validate_complete_event_log,
)

MODEL_SPACE_GAP = "Model-space RGB/float parity is not verified yet"
PRODUCT_OUTPUT_GAP = "Product-output PSNR/SSIM and visual crops are not generated yet"
PROFILE_PARAMETER_KEYS = (
    "mode",
    "vspipe_requests",
    "num_streams",
    "vapoursynth_threads",
    "cuda_graph",
)


class CampaignError(RuntimeError):
    """Raised when campaign artifacts cannot form one comparable result."""


@dataclass(frozen=True)
class StabilityAssessment:
    """Result of applying the campaign's two-phase stability policy."""

    status: str
    threshold: float
    full_relative_spread: float
    consensus_rounds: tuple[int, ...] = ()
    consensus_values_fps: tuple[float, ...] = ()
    consensus_relative_spread: float | None = None
    outlier_round: int | None = None
    outlier_fps: float | None = None

    def as_dict(self) -> dict[str, Any]:
        consensus = None
        if self.consensus_rounds:
            consensus = {
                "required_rounds": 4,
                "rounds": list(self.consensus_rounds),
                "values_fps": list(self.consensus_values_fps),
                "relative_spread": self.consensus_relative_spread,
                "accepted": self.consensus_relative_spread is not None
                and self.consensus_relative_spread <= self.threshold,
            }
        outlier = None
        if self.outlier_round is not None:
            outlier = {
                "round": self.outlier_round,
                "fps": self.outlier_fps,
            }
        return {
            "status": self.status,
            "policy": "full-range-3-then-consensus-4-of-5",
            "threshold": self.threshold,
            "full_relative_spread": self.full_relative_spread,
            "consensus": consensus,
            "outlier": outlier,
        }


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


def _cpu_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("measured", {}).get("metrics", {}).get("cpu")
    if not isinstance(value, dict):
        raise CampaignError("Manifest has no measured CPU accounting")
    contract = {
        "accounting": value.get("accounting"),
        "scope": value.get("scope"),
        "available_logical_cpus": value.get("available_logical_cpus"),
    }
    if not isinstance(contract["accounting"], str) or not contract["accounting"]:
        raise CampaignError("Manifest has no CPU accounting method")
    if not isinstance(contract["scope"], str) or not contract["scope"]:
        raise CampaignError("Manifest has no CPU accounting scope")
    if (
        not isinstance(contract["available_logical_cpus"], int)
        or contract["available_logical_cpus"] <= 0
    ):
        raise CampaignError("Manifest has an invalid available logical CPU count")
    return contract


def _lifecycle_contract(manifest: dict[str, Any]) -> dict[str, str]:
    value = manifest.get("measured", {}).get("metrics", {}).get("lifecycle")
    if not isinstance(value, dict):
        raise CampaignError("Manifest has no lifecycle timing scopes")
    clock = value.get("clock")
    boundary_contract = value.get("boundary_contract")
    if (
        not isinstance(clock, str)
        or not clock
        or not isinstance(boundary_contract, str)
        or not boundary_contract
    ):
        raise CampaignError("Manifest has an invalid lifecycle timing contract")
    wall_time_sec = _metric(manifest, "wall_time_sec")
    total_sec = _metric(manifest, "lifecycle", "total_sec")
    if abs(total_sec - wall_time_sec) > 0.001:
        raise CampaignError("Lifecycle scopes do not cover measured wall time")
    scope_total = sum(
        _metric(manifest, "lifecycle", key)
        for key in (
            "startup_sec",
            "steady_state_frame_loop_sec",
            "finalize_mux_sec",
        )
    )
    if abs(scope_total - total_sec) > 0.001:
        raise CampaignError("Lifecycle scope durations do not sum to total time")
    return {
        "clock": clock,
        "boundary_contract": boundary_contract,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _relative_spread(values: list[float]) -> float:
    median = _median(values)
    if median <= 0:
        raise CampaignError("FPS values must have a positive median")
    return (max(values) - min(values)) / median


def _assess_stability(
    values: list[float],
    *,
    threshold: float,
) -> StabilityAssessment:
    """Apply the canonical three-run, then four-of-five consensus policy."""
    if len(values) not in {3, 5}:
        raise CampaignError(
            f"Stability assessment requires 3 or 5 values, got {len(values)}"
        )

    full_spread = _relative_spread(values)
    if full_spread <= threshold:
        return StabilityAssessment(
            status="stable",
            threshold=threshold,
            full_relative_spread=full_spread,
        )
    if len(values) == 3:
        return StabilityAssessment(
            status="needs-extra-runs",
            threshold=threshold,
            full_relative_spread=full_spread,
        )

    indexed_values = tuple(enumerate(values, start=1))
    candidates = []
    for candidate in combinations(indexed_values, 4):
        candidate_values = [value for _, value in candidate]
        candidates.append((_relative_spread(candidate_values), candidate))
    consensus_spread, consensus = min(
        candidates,
        key=lambda item: (item[0], tuple(index for index, _ in item[1])),
    )
    consensus_rounds = tuple(index for index, _ in consensus)
    consensus_values = tuple(value for _, value in consensus)
    excluded = next(item for item in indexed_values if item not in consensus)
    status = (
        "stable-with-one-outlier"
        if consensus_spread <= threshold
        else "unstable"
    )
    return StabilityAssessment(
        status=status,
        threshold=threshold,
        full_relative_spread=full_spread,
        consensus_rounds=consensus_rounds,
        consensus_values_fps=consensus_values,
        consensus_relative_spread=consensus_spread,
        outlier_round=excluded[0] if status == "stable-with-one-outlier" else None,
        outlier_fps=excluded[1] if status == "stable-with-one-outlier" else None,
    )


def _validate_model_space_report(
    path: Path,
    *,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    report = _load_json(path)
    checks = {
        "document type": (report.get("document_type"), "model-space-parity"),
        "status": (report.get("status"), "valid"),
        "publishable": (report.get("publishable"), True),
        "workload": (report.get("workload_id"), contract["workload_id"]),
        "variant": (report.get("variant"), contract["variant"]),
        "input SHA256": (
            report.get("assets", {}).get("input_sha256"),
            contract["input_sha256"],
        ),
        "ONNX SHA256": (
            report.get("assets", {}).get("onnx_sha256"),
            contract["onnx_sha256"],
        ),
        "frame indices": (
            report.get("frame_indices"),
            contract["model_space_frame_indices"],
        ),
        "reference engine": (
            report.get("reference", {}).get("engine_sha256"),
            contract["engine_hashes"]["ai-media"],
        ),
        "reference image": (
            report.get("reference", {}).get("image", {}).get("id"),
            contract["image_ids"]["ai-media"],
        ),
        "reference revision": (
            report.get("reference", {})
            .get("image", {})
            .get("repository_revision"),
            contract["repository_revision"],
        ),
        "reference source state": (
            str(
                report.get("reference", {})
                .get("image", {})
                .get("source_dirty")
            ),
            "0",
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CampaignError(f"Model-space report changed {label}")

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise CampaignError("Model-space report has no comparisons")
    by_implementation = {
        comparison.get("implementation"): comparison
        for comparison in comparisons
        if isinstance(comparison, dict)
    }
    expected_contracts = {
        "vs-mlrt": (
            contract["engine_hashes"]["vstrt"],
            contract["image_ids"]["vstrt"],
        ),
        "VSGAN-tensorrt-docker": (
            contract["engine_hashes"]["vsgan"],
            contract["image_ids"]["vsgan"],
        ),
    }
    for implementation, (engine_sha256, image_id) in expected_contracts.items():
        comparison = by_implementation.get(implementation)
        if not isinstance(comparison, dict):
            raise CampaignError(
                f"Model-space report has no {implementation} comparison"
            )
        if comparison.get("status") != "valid":
            raise CampaignError(
                f"Model-space report marks {implementation} as invalid"
            )
        if comparison.get("engine_sha256") != engine_sha256:
            raise CampaignError(
                f"Model-space report changed {implementation} engine"
            )
        image = comparison.get("image", {})
        if not isinstance(image, dict):
            raise CampaignError(
                f"Model-space report has no {implementation} image identity"
            )
        image_checks = {
            "image": (image.get("id"), image_id),
            "revision": (
                image.get("repository_revision"),
                contract["repository_revision"],
            ),
            "source state": (str(image.get("source_dirty")), "0"),
        }
        for label, (actual, expected) in image_checks.items():
            if actual != expected:
                raise CampaignError(
                    f"Model-space report changed {implementation} {label}"
                )
    return {
        "status": "valid",
        "report": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
    }


def _validate_report_artifact(
    value: dict[str, Any],
    *,
    root: Path,
    path_key: str,
    hash_key: str,
    label: str,
) -> Path:
    relative_path = value.get(path_key)
    checksum = value.get(hash_key)
    if not isinstance(relative_path, str) or not relative_path:
        raise CampaignError(f"{label} has no artifact path")
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise CampaignError(f"{label} artifact path escapes the repository")
    if not path.is_file():
        raise CampaignError(f"{label} artifact is missing: {path}")
    if checksum != sha256_file(path):
        raise CampaignError(f"{label} artifact SHA256 changed")
    return path


def _validate_quality_run_manifest(
    path: Path,
    *,
    implementation: str,
    product: str,
    contract: dict[str, Any],
) -> None:
    manifest = _load_json(path)
    image = manifest.get("environment", {}).get("image", {})
    checks = {
        "status": (manifest.get("status"), "valid"),
        "product": (manifest.get("product"), product),
        "workload": (manifest.get("workload_id"), contract["workload_id"]),
        "variant": (manifest.get("variant"), contract["variant"]),
        "frame count": (
            manifest.get("parameters", {}).get("frames"),
            contract["frames"],
        ),
        "encoder contract": (
            manifest.get("parameters", {}).get("encoder"),
            contract["encoder"],
        ),
        "input SHA256": (_asset_sha(manifest, "input"), contract["input_sha256"]),
        "ONNX SHA256": (_asset_sha(manifest, "onnx"), contract["onnx_sha256"]),
        "engine SHA256": (
            _asset_sha(manifest, "engine"),
            contract["engine_hashes"][implementation],
        ),
        "image": (image.get("id"), contract["image_ids"][implementation]),
        "revision": (
            image.get("repository_revision"),
            contract["repository_revision"],
        ),
        "source state": (str(image.get("source_dirty")), "0"),
        "reproducibility": (
            manifest.get("reproducibility", {}).get("publishable"),
            True,
        ),
        "output validation": (
            manifest.get("measured", {}).get("validation", {}).get("valid"),
            True,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CampaignError(
                f"Product-output {product} run changed {label}"
            )
    if implementation in {"vstrt", "vsgan"}:
        profile = _execution_profile_contract(
            manifest,
            implementation=implementation,
            expected_profile=contract["execution_profile"],
        )
        if profile != contract["execution_profiles"][implementation]:
            raise CampaignError(
                f"Product-output {product} run changed execution profile"
            )


def _validate_product_output_report(
    path: Path,
    *,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    report = _load_json(path)
    checks = {
        "document type": (report.get("document_type"), "product-output-parity"),
        "status": (report.get("status"), "valid"),
        "publishable": (report.get("publishable"), True),
        "workload": (report.get("workload_id"), contract["workload_id"]),
        "variant": (report.get("variant"), contract["variant"]),
        "input SHA256": (
            report.get("assets", {}).get("input_sha256"),
            contract["input_sha256"],
        ),
        "ONNX SHA256": (
            report.get("assets", {}).get("onnx_sha256"),
            contract["onnx_sha256"],
        ),
        "frame indices": (
            report.get("frame_indices"),
            contract["product_output_frame_indices"],
        ),
        "thresholds": (
            report.get("thresholds"),
            contract["product_output_thresholds"],
        ),
        "reference engine": (
            report.get("reference", {}).get("engine_sha256"),
            contract["engine_hashes"]["ai-media"],
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CampaignError(f"Product-output report changed {label}")

    reference = report.get("reference")
    if not isinstance(reference, dict):
        raise CampaignError("Product-output report has no reference")
    reference_manifest = _validate_report_artifact(
        reference,
        root=root,
        path_key="run_manifest",
        hash_key="run_manifest_sha256",
        label="Product-output reference",
    )
    _validate_quality_run_manifest(
        reference_manifest,
        implementation="ai-media",
        product="ai-media-enhancer",
        contract=contract,
    )

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise CampaignError("Product-output report has no comparisons")
    by_implementation = {
        comparison.get("implementation"): comparison
        for comparison in comparisons
        if isinstance(comparison, dict)
    }
    expected_contracts = {
        "vs-mlrt": ("vstrt", contract["engine_hashes"]["vstrt"]),
        "VSGAN-tensorrt-docker": ("vsgan", contract["engine_hashes"]["vsgan"]),
    }
    for implementation, (
        campaign_implementation,
        engine_sha256,
    ) in expected_contracts.items():
        comparison = by_implementation.get(implementation)
        if not isinstance(comparison, dict):
            raise CampaignError(
                f"Product-output report has no {implementation} comparison"
            )
        if comparison.get("status") != "valid":
            raise CampaignError(
                f"Product-output report marks {implementation} as invalid"
            )
        if comparison.get("engine_sha256") != engine_sha256:
            raise CampaignError(
                f"Product-output report changed {implementation} engine"
            )
        run_manifest = _validate_report_artifact(
            comparison,
            root=root,
            path_key="run_manifest",
            hash_key="run_manifest_sha256",
            label=f"Product-output {implementation}",
        )
        _validate_quality_run_manifest(
            run_manifest,
            implementation=campaign_implementation,
            product=implementation,
            contract=contract,
        )
        metrics = comparison.get("metrics")
        if not isinstance(metrics, dict):
            raise CampaignError(
                f"Product-output report has no {implementation} metrics"
            )
        for metric_name in ("psnr", "ssim"):
            metric = metrics.get(metric_name)
            if not isinstance(metric, dict):
                raise CampaignError(
                    f"Product-output report has no {implementation} {metric_name}"
                )
            _validate_report_artifact(
                metric,
                root=root,
                path_key="stats_path",
                hash_key="stats_sha256",
                label=f"Product-output {implementation} {metric_name} stats",
            )
            _validate_report_artifact(
                metric,
                root=root,
                path_key="ffmpeg_log",
                hash_key="ffmpeg_log_sha256",
                label=f"Product-output {implementation} {metric_name} log",
            )

    visual_crops = report.get("visual_crops")
    if not isinstance(visual_crops, dict):
        raise CampaignError("Product-output report has no visual crops")
    expected_products = {
        "ai-media-enhancer",
        "vs-mlrt",
        "VSGAN-tensorrt-docker",
    }
    if set(visual_crops) != expected_products:
        raise CampaignError("Product-output visual crop implementations differ")
    expected_crop_count = len(contract["product_output_frame_indices"]) * len(
        contract["product_output_crop_names"]
    )
    for implementation, crops in visual_crops.items():
        if not isinstance(crops, list) or len(crops) != expected_crop_count:
            raise CampaignError(
                f"Product-output {implementation} visual crop set is incomplete"
            )
        actual_keys = {
            (crop.get("frame_index"), crop.get("crop"))
            for crop in crops
            if isinstance(crop, dict)
        }
        expected_keys = {
            (frame_index, crop_name)
            for frame_index in contract["product_output_frame_indices"]
            for crop_name in contract["product_output_crop_names"]
        }
        if actual_keys != expected_keys:
            raise CampaignError(
                f"Product-output {implementation} visual crop contract changed"
            )
        for crop in crops:
            _validate_report_artifact(
                crop,
                root=root,
                path_key="path",
                hash_key="sha256",
                label=f"Product-output {implementation} visual crop",
            )

    return {
        "status": "valid",
        "report": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
    }


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


def _execution_profile_contract(
    manifest: dict[str, Any],
    *,
    implementation: str,
    expected_profile: str,
) -> dict[str, Any]:
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict):
        raise CampaignError(f"{implementation} manifest has no parameters")
    missing = [key for key in PROFILE_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise CampaignError(
            f"{implementation} manifest has no execution profile fields: "
            + ", ".join(missing)
        )
    profile = {key: parameters[key] for key in PROFILE_PARAMETER_KEYS}
    if profile["mode"] != expected_profile:
        raise CampaignError(
            f"{implementation} execution profile is {profile['mode']!r}, "
            f"expected {expected_profile!r}"
        )
    expected_class = (
        "parity"
        if expected_profile == "parity" and implementation == "vstrt"
        else (
            "single-stream-parity"
            if expected_profile == "parity"
            else expected_profile
        )
    )
    if manifest.get("comparison_class") != expected_class:
        raise CampaignError(
            f"{implementation} comparison class does not match "
            f"{expected_profile} execution profile"
        )
    return profile


def _validate_common_contract(
    rounds: list[dict[str, dict[str, Any]]],
    *,
    root: Path,
    idle_seconds: float,
    execution_profile: str,
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
    cpu_contract = _cpu_contract(first)
    lifecycle_contract = _lifecycle_contract(first)
    if not isinstance(encoder, dict):
        raise CampaignError("Campaign manifests have no exact encoder contract")
    expected_revision = os.environ.get("AI_MEDIA_BUILD_REVISION")
    if expected_revision and expected_revision != "unknown" and revision != expected_revision:
        raise CampaignError(
            "Campaign repository revision does not match the aggregator image"
        )

    image_ids: dict[str, str] = {}
    engine_hashes: dict[str, str] = {}
    execution_profiles: dict[str, dict[str, Any]] = {}
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
                "CPU accounting": (_cpu_contract(manifest), cpu_contract),
                "lifecycle timing": (
                    _lifecycle_contract(manifest),
                    lifecycle_contract,
                ),
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
            if implementation in {"vstrt", "vsgan"}:
                profile = _execution_profile_contract(
                    manifest,
                    implementation=implementation,
                    expected_profile=execution_profile,
                )
                previous_profile = execution_profiles.setdefault(
                    implementation,
                    profile,
                )
                if profile != previous_profile:
                    raise CampaignError(
                        f"{implementation} execution profile changed between rounds"
                    )

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
        "model_space_frame_indices": workload.get("quality", {})
        .get("model_space", {})
        .get("frame_indices"),
        "product_output_frame_indices": workload.get("quality", {})
        .get("product_output", {})
        .get("frame_indices"),
        "product_output_thresholds": workload.get("quality", {})
        .get("product_output", {})
        .get("thresholds"),
        "product_output_crop_names": [
            crop.get("name")
            for crop in workload.get("quality", {})
            .get("product_output", {})
            .get("crops", [])
            if isinstance(crop, dict)
        ],
        "cpu_accounting": cpu_contract,
        "lifecycle_timing": lifecycle_contract,
        "image_ids": image_ids,
        "engine_hashes": engine_hashes,
        "execution_profile": execution_profile,
        "execution_profiles": execution_profiles,
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
            "median_cpu_cores": _median(
                [
                    _metric(manifest, "cpu", "average_cores")
                    for manifest in manifests
                ]
            ),
            "median_cpu_capacity_percent": _median(
                [
                    _metric(manifest, "cpu", "capacity_percent")
                    for manifest in manifests
                ]
            ),
            "median_startup_sec": _median(
                [
                    _metric(manifest, "lifecycle", "startup_sec")
                    for manifest in manifests
                ]
            ),
            "median_steady_state_frame_loop_sec": _median(
                [
                    _metric(
                        manifest,
                        "lifecycle",
                        "steady_state_frame_loop_sec",
                    )
                    for manifest in manifests
                ]
            ),
            "median_finalize_mux_sec": _median(
                [
                    _metric(manifest, "lifecycle", "finalize_mux_sec")
                    for manifest in manifests
                ]
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
        f"Status: `{summary['status']}`. Publication ready: "
        f"`{'yes' if summary['publication']['ready'] else 'no'}`.",
        f"Execution profile: `{summary['comparison_profile']}`.",
        "",
        "| Implementation | Runs | Median FPS | vs ai-media | Median wall, s | "
        "CPU cores | CPU capacity, % | GPU util, % | Power, W | J/frame | "
        "Peak VRAM, MiB | Bitrate, Mbps | Size, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        lines.append(
            f"| {result['product']} | {summary['parameters']['rounds']} | "
            f"{stats['median_fps']:.3f} | {result['relative_to_ai_media_percent']:+.2f}% | "
            f"{stats['median_wall_time_sec']:.2f} | "
            f"{stats['median_cpu_cores']:.3f} | "
            f"{stats['median_cpu_capacity_percent']:.2f} | "
            f"{stats['median_gpu_utilization_percent']:.2f} | "
            f"{stats['median_power_w']:.2f} | "
            f"{stats['median_joules_per_frame']:.3f} | "
            f"{stats['median_peak_vram_mib']:.1f} | "
            f"{stats['median_output_bitrate_mbps']:.3f} | "
            f"{stats['median_output_size_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Implementation | Startup, s | Steady-state frame loop, s | "
            "Finalize + mux, s |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        lines.append(
            f"| {result['product']} | {stats['median_startup_sec']:.3f} | "
            f"{stats['median_steady_state_frame_loop_sec']:.3f} | "
            f"{stats['median_finalize_mux_sec']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Implementation | Stability | Full spread | 4-of-5 spread | "
            "Outlier | Raw FPS |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        stability = result["stability"]
        consensus = stability["consensus"]
        outlier = stability["outlier"]
        consensus_spread = (
            f"{consensus['relative_spread']:.2%}" if consensus else "-"
        )
        outlier_label = (
            f"round {outlier['round']}: {outlier['fps']:.3f} FPS"
            if outlier
            else "-"
        )
        raw_values = ", ".join(f"{value:.3f}" for value in stats["values_fps"])
        lines.append(
            f"| {result['product']} | {stability['status']} | "
            f"{stability['full_relative_spread']:.2%} | {consensus_spread} | "
            f"{outlier_label} | {raw_values} |"
        )
    if summary["publication"]["warnings"]:
        lines.extend(["", "Publication warnings:"])
        lines.extend(
            f"- {warning}" for warning in summary["publication"]["warnings"]
        )
    if summary["publication"]["errors"]:
        lines.extend(["", "Publication gaps:"])
        lines.extend(f"- {gap}" for gap in summary["publication"]["errors"])
    lines.append("")
    return "\n".join(lines)


def aggregate_campaign(
    campaign_dir: Path,
    *,
    root: Path,
    idle_seconds: float,
    execution_profile: str = "parity",
    model_space_report: Path | None = None,
    product_output_report: Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate all completed rounds in one campaign directory."""
    if execution_profile not in EXECUTION_PROFILES:
        raise CampaignError(f"Unknown campaign execution profile: {execution_profile}")
    try:
        campaign_config = load_campaign_config(campaign_dir / CONFIG_NAME)
    except CampaignEventError as exc:
        raise CampaignError(f"Invalid campaign config: {exc}") from exc
    if campaign_config.execution_profile != execution_profile:
        raise CampaignError(
            "Campaign config execution profile does not match aggregation request"
        )
    rounds = _load_rounds(campaign_dir)
    events_path = campaign_dir / EVENT_LOG_NAME
    try:
        events = validate_complete_event_log(
            load_events(events_path),
            rounds=len(rounds),
            idle_seconds=idle_seconds,
        )
    except CampaignEventError as exc:
        raise CampaignError(f"Invalid campaign execution log: {exc}") from exc
    workload, contract = _validate_common_contract(
        rounds,
        root=root,
        idle_seconds=idle_seconds,
        execution_profile=execution_profile,
    )
    quality: dict[str, Any] = {}
    publication_errors = [MODEL_SPACE_GAP, PRODUCT_OUTPUT_GAP]
    if model_space_report is not None:
        quality["model_space"] = _validate_model_space_report(
            model_space_report,
            contract=contract,
            root=root,
        )
        publication_errors.remove(MODEL_SPACE_GAP)
    if product_output_report is not None:
        quality["product_output"] = _validate_product_output_report(
            product_output_report,
            contract=contract,
            root=root,
        )
        publication_errors.remove(PRODUCT_OUTPUT_GAP)
    implementation_results: dict[str, Any] = {}
    spread_threshold = float(workload["benchmark"]["spread_threshold"])
    for name, product in IMPLEMENTATIONS.items():
        statistics_report = _implementation_statistics(rounds, name)
        stability = _assess_stability(
            statistics_report["values_fps"],
            threshold=spread_threshold,
        )
        implementation_results[name] = {
            "product": product,
            "image_id": contract["image_ids"][name],
            "engine_sha256": contract["engine_hashes"][name],
            "statistics": statistics_report,
            "stability": stability.as_dict(),
        }

    ai_fps = implementation_results["ai-media"]["statistics"]["median_fps"]
    for result in implementation_results.values():
        median_fps = result["statistics"]["median_fps"]
        result["relative_to_ai_media_percent"] = (median_fps / ai_fps - 1) * 100

    unstable = [
        name
        for name, result in implementation_results.items()
        if result["stability"]["status"] in {"needs-extra-runs", "unstable"}
    ]
    stable_with_outlier = [
        name
        for name, result in implementation_results.items()
        if result["stability"]["status"] == "stable-with-one-outlier"
    ]
    needs_extra = len(rounds) == 3 and bool(unstable)
    status = "needs-extra-runs" if needs_extra else ("unstable" if unstable else "valid")
    publication_warnings = []
    for name in stable_with_outlier:
        result = implementation_results[name]
        stability = result["stability"]
        outlier = stability["outlier"]
        consensus = stability["consensus"]
        publication_warnings.append(
            f"{result['product']} is stable with one outlier: "
            f"round {outlier['round']} at {outlier['fps']:.3f} FPS; "
            f"full spread {stability['full_relative_spread']:.2%}, "
            f"4-of-5 consensus spread {consensus['relative_spread']:.2%}"
        )
    publication_ready = status == "valid" and not publication_errors
    summary = {
        "schema_version": 1,
        "document_type": "benchmark-campaign",
        "status": status,
        "scope": "rotated-campaign",
        "comparison_profile": execution_profile,
        "publishable": publication_ready,
        "publication": {
            "ready": publication_ready,
            "errors": publication_errors,
            "warnings": publication_warnings,
        },
        "workload_id": contract["workload_id"],
        "variant": contract["variant"],
        "parameters": {
            "rounds": len(rounds),
            "warmup_frames": contract["warmup_frames"],
            "measured_frames": contract["frames"],
            "idle_seconds": idle_seconds,
            "spread_threshold": spread_threshold,
            "stability_policy": "full-range-3-then-consensus-4-of-5",
            "encoder": contract["encoder"],
            "cpu_accounting": contract["cpu_accounting"],
            "execution_profiles": contract["execution_profiles"],
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
        "execution": {
            "config": relative_artifact_path(campaign_dir / CONFIG_NAME, root),
            "config_sha256": sha256_file(campaign_dir / CONFIG_NAME),
            "runner_arguments": {
                "vstrt": campaign_config.vstrt_arguments,
                "vsgan": campaign_config.vsgan_arguments,
            },
            "event_log": relative_artifact_path(events_path, root),
            "event_log_sha256": sha256_file(events_path),
        },
        "quality": quality,
        "rounds": [
            {
                "index": index,
                "order": [
                    event.implementation
                    for event in events
                    if event.round_index == index
                ],
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
        "stable_with_outlier_implementations": stable_with_outlier,
        "needs_extra_runs": needs_extra,
        "implementations": implementation_results,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate a rotated benchmark campaign")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--root", default="/app")
    parser.add_argument("--idle-seconds", type=float, required=True)
    parser.add_argument(
        "--execution-profile",
        choices=EXECUTION_PROFILES,
        required=True,
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--model-space-report",
        default=None,
        help="Optional valid model-space parity report for this exact campaign",
    )
    parser.add_argument(
        "--product-output-report",
        default=None,
        help="Optional valid product-output parity report for this exact campaign",
    )
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
            execution_profile=args.execution_profile,
            model_space_report=(
                Path(args.model_space_report) if args.model_space_report else None
            ),
            product_output_report=(
                Path(args.product_output_report)
                if args.product_output_report
                else None
            ),
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
