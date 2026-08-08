#!/usr/bin/env python3
"""Aggregate a rotated multi-product benchmark campaign."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

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
from benchmarks.scripts.campaign.report import render_markdown
from benchmarks.scripts.contracts.manifest import (
    ManifestContractError as CampaignError,
)
from benchmarks.scripts.contracts.manifest import (
    RunExpectation,
    RunIdentity,
    extract_run_identity,
    load_json,
    validate_execution_profile,
    validate_run_manifest,
)
from benchmarks.scripts.contracts.model_space import (
    TensorComparisonExpectation,
    TensorReportExpectation,
    validate_inference_report,
    validate_preprocessing_report,
)
from benchmarks.scripts.runtime.environment import (
    relative_artifact_path,
    sha256_file,
    write_json,
)
from benchmarks.scripts.runtime.suite import compute_suite_statistics
from trtvideo.benchmarking.lifecycle import median_detailed_phase_intervals

INFERENCE_PARITY_GAP = "Shared-input TensorRT inference parity is not verified yet"
PREPROCESSING_DIAGNOSTIC_GAP = "Production preprocessing diagnostics are not recorded yet"
PRODUCT_OUTPUT_GAP = "Product-output PSNR/SSIM and visual crops are not generated yet"


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


def _manifest_path(campaign_dir: Path, implementation: str, round_index: int) -> Path:
    return campaign_dir / implementation / f"round-{round_index:02d}" / "run-01" / "manifest.json"


def _load_rounds(campaign_dir: Path) -> list[dict[str, dict[str, Any]]]:
    rounds: list[dict[str, dict[str, Any]]] = []
    missing_started = False
    for round_index in ROUND_ORDERS:
        paths = {name: _manifest_path(campaign_dir, name, round_index) for name in IMPLEMENTATIONS}
        present = {name: path.is_file() for name, path in paths.items()}
        if not any(present.values()):
            missing_started = True
            continue
        if missing_started:
            raise CampaignError("Campaign rounds are not contiguous")
        missing = [name for name, exists in present.items() if not exists]
        if missing:
            raise CampaignError(f"Round {round_index} is incomplete; missing: {', '.join(missing)}")
        rounds.append({name: load_json(path) for name, path in paths.items()})
    if len(rounds) not in {3, 5}:
        raise CampaignError(f"Campaign requires 3 or 5 complete rounds, got {len(rounds)}")
    return rounds


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
        raise CampaignError(f"Stability assessment requires 3 or 5 values, got {len(values)}")

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
    status = "stable-with-one-outlier" if consensus_spread <= threshold else "unstable"
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


def _tensor_report_expectation(contract: dict[str, Any]) -> TensorReportExpectation:
    return TensorReportExpectation(
        workload_id=contract["workload_id"],
        variant=contract["variant"],
        contract_version=contract["model_space_contract_version"],
        input_sha256=contract["input_sha256"],
        onnx_sha256=contract["onnx_sha256"],
        comparisons=(
            TensorComparisonExpectation(
                implementation="vs-mlrt",
                engine_sha256=contract["engine_hashes"]["vstrt"],
                image_id=contract["image_ids"]["vstrt"],
                repository_revision=contract["repository_revision"],
                execution_profile=contract["execution_profiles"]["vstrt"],
            ),
            TensorComparisonExpectation(
                implementation="VSGAN-tensorrt-docker",
                engine_sha256=contract["engine_hashes"]["vsgan"],
                image_id=contract["image_ids"]["vsgan"],
                repository_revision=contract["repository_revision"],
                execution_profile=contract["execution_profiles"]["vsgan"],
            ),
        ),
        execution_profile=contract["execution_profile"],
        frame_indices=contract["model_space_frame_indices"],
        reference_engine_sha256=contract["engine_hashes"]["trtvideo"],
        reference_image_id=contract["image_ids"]["trtvideo"],
        reference_revision=contract["repository_revision"],
        reference_source_dirty="0",
        reference_execution_profile={
            "execution_profile": contract["execution_profile"],
            "cuda_graph": False,
        },
    )


def _validate_inference_report(
    path: Path,
    *,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    validate_inference_report(
        load_json(path),
        expectation=_tensor_report_expectation(contract),
    )
    return {
        "status": "valid",
        "acceptance_gate": True,
        "report": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
    }


def _validate_preprocessing_report(
    path: Path,
    *,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    validate_preprocessing_report(
        load_json(path),
        expectation=_tensor_report_expectation(contract),
    )
    return {
        "status": "complete",
        "acceptance_gate": False,
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
    manifest = load_json(path)
    expected_profile = (
        contract["execution_profiles"][implementation]
        if implementation in {"vstrt", "vsgan"}
        else None
    )
    identity = validate_run_manifest(
        manifest,
        expectation=RunExpectation(
            product=product,
            workload_id=contract["workload_id"],
            variant=contract["variant"],
            benchmark_contract_version=contract["benchmark_contract_version"],
            implementation=(implementation if expected_profile is not None else None),
            execution_profile=expected_profile,
            require_media_validation=True,
            require_workload_identity=False,
            require_warmup_frames=False,
        ),
    )
    checks = {
        "frame count": (identity.frames, contract["product_output_frames"]),
        "encoder contract": (identity.encoder, contract["encoder"]),
        "input SHA256": (identity.input_sha256, contract["input_sha256"]),
        "ONNX SHA256": (identity.onnx_sha256, contract["onnx_sha256"]),
        "engine SHA256": (
            identity.engine_sha256,
            contract["engine_hashes"][implementation],
        ),
        "image": (identity.image_id, contract["image_ids"][implementation]),
        "revision": (
            identity.repository_revision,
            contract["repository_revision"],
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CampaignError(f"Product-output {product} run changed {label}")


def _validate_product_output_report(
    path: Path,
    *,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    report = load_json(path)
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
            contract["engine_hashes"]["trtvideo"],
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
        implementation="trtvideo",
        product="trtvideo",
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
            raise CampaignError(f"Product-output report has no {implementation} comparison")
        if comparison.get("status") != "valid":
            raise CampaignError(f"Product-output report marks {implementation} as invalid")
        if comparison.get("engine_sha256") != engine_sha256:
            raise CampaignError(f"Product-output report changed {implementation} engine")
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
            raise CampaignError(f"Product-output report has no {implementation} metrics")
        for metric_name in ("psnr", "ssim"):
            metric = metrics.get(metric_name)
            if not isinstance(metric, dict):
                raise CampaignError(f"Product-output report has no {implementation} {metric_name}")
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
        "trtvideo",
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
            raise CampaignError(f"Product-output {implementation} visual crop set is incomplete")
        actual_keys = {
            (crop.get("frame_index"), crop.get("crop")) for crop in crops if isinstance(crop, dict)
        }
        expected_keys = {
            (frame_index, crop_name)
            for frame_index in contract["product_output_frame_indices"]
            for crop_name in contract["product_output_crop_names"]
        }
        if actual_keys != expected_keys:
            raise CampaignError(f"Product-output {implementation} visual crop contract changed")
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
) -> RunIdentity:
    try:
        return validate_run_manifest(
            manifest,
            expectation=RunExpectation(
                product=IMPLEMENTATIONS[implementation],
                run_index=1,
            ),
        )
    except CampaignError as exc:
        raise CampaignError(f"{implementation} round {round_index}: {exc}") from exc


def _validate_common_contract(
    rounds: list[dict[str, dict[str, Any]]],
    *,
    root: Path,
    idle_seconds: float,
    execution_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = rounds[0]["trtvideo"]
    first_identity = extract_run_identity(first)
    workload_id = first_identity.workload_id
    variant = first_identity.variant
    revision = first_identity.repository_revision
    encoder = first_identity.encoder
    input_sha = first_identity.input_sha256
    onnx_sha = first_identity.onnx_sha256
    gpu = first.get("environment", {}).get("gpu")
    cpu = first.get("environment", {}).get("cpu")
    frames = first_identity.frames
    warmup_frames = first_identity.warmup_frames
    contract_version = first_identity.benchmark_contract_version
    cpu_contract = _cpu_contract(first)
    lifecycle_contract = _lifecycle_contract(first)
    expected_revision = os.environ.get("TRTVIDEO_BUILD_REVISION")
    if expected_revision and expected_revision != "unknown" and revision != expected_revision:
        raise CampaignError("Campaign repository revision does not match the aggregator image")

    image_ids: dict[str, str] = {}
    engine_hashes: dict[str, str] = {}
    execution_profiles: dict[str, dict[str, Any]] = {}
    for round_index, round_data in enumerate(rounds, start=1):
        for implementation, manifest in round_data.items():
            identity = _validate_manifest(
                manifest,
                implementation=implementation,
                round_index=round_index,
            )
            environment = manifest.get("environment", {})
            checks = {
                "workload": (identity.workload_id, workload_id),
                "variant": (identity.variant, variant),
                "repository revision": (
                    identity.repository_revision,
                    revision,
                ),
                "GPU": (environment.get("gpu"), gpu),
                "CPU": (environment.get("cpu"), cpu),
                "CPU accounting": (_cpu_contract(manifest), cpu_contract),
                "lifecycle timing": (
                    _lifecycle_contract(manifest),
                    lifecycle_contract,
                ),
                "frame count": (identity.frames, frames),
                "warmup frame count": (identity.warmup_frames, warmup_frames),
                "benchmark contract version": (
                    identity.benchmark_contract_version,
                    contract_version,
                ),
                "encoder contract": (identity.encoder, encoder),
                "input SHA256": (identity.input_sha256, input_sha),
                "ONNX SHA256": (identity.onnx_sha256, onnx_sha),
                "workload manifest SHA256": (
                    identity.workload_sha256,
                    first_identity.workload_sha256,
                ),
            }
            for label, (actual, expected) in checks.items():
                if actual != expected:
                    raise CampaignError(f"{implementation} round {round_index} changed {label}")
            image_id = identity.image_id
            previous_image = image_ids.setdefault(implementation, image_id)
            if image_id != previous_image:
                raise CampaignError(f"{implementation} image changed between rounds")
            engine_hash = identity.engine_sha256
            previous_engine = engine_hashes.setdefault(implementation, engine_hash)
            if engine_hash != previous_engine:
                raise CampaignError(f"{implementation} engine changed between rounds")
            if implementation in {"vstrt", "vsgan"}:
                profile = validate_execution_profile(
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

    if engine_hashes["trtvideo"] != engine_hashes["vstrt"]:
        raise CampaignError("trtvideo and vstrt must use the same serialized engine")

    workload_asset = first.get("assets", {}).get("workload_manifest", {})
    workload_path = root / str(workload_asset.get("path", ""))
    if not workload_path.is_file():
        raise CampaignError(f"Canonical workload manifest not found: {workload_path}")
    if sha256_file(workload_path) != workload_asset.get("sha256"):
        raise CampaignError("Canonical workload manifest SHA256 changed")
    workload = load_json(workload_path)
    benchmark = workload.get("benchmark", {})
    canonical_checks = {
        "contract version": (
            contract_version,
            benchmark.get("contract_version"),
        ),
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
        "benchmark_contract_version": contract_version,
        "product_output_frames": workload.get("clip", {}).get("frames"),
        "model_space_frame_indices": workload.get("quality", {})
        .get("model_space", {})
        .get("frame_indices"),
        "model_space_contract_version": workload.get("quality", {})
        .get("model_space", {})
        .get("contract_version"),
        "product_output_frame_indices": workload.get("quality", {})
        .get("product_output", {})
        .get("frame_indices"),
        "product_output_thresholds": workload.get("quality", {})
        .get("product_output", {})
        .get("thresholds"),
        "product_output_crop_names": [
            crop.get("name")
            for crop in workload.get("quality", {}).get("product_output", {}).get("crops", [])
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
                [_metric(manifest, "cpu", "average_cores") for manifest in manifests]
            ),
            "median_cpu_capacity_percent": _median(
                [_metric(manifest, "cpu", "capacity_percent") for manifest in manifests]
            ),
            "median_startup_sec": _median(
                [_metric(manifest, "lifecycle", "startup_sec") for manifest in manifests]
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
                [_metric(manifest, "lifecycle", "finalize_mux_sec") for manifest in manifests]
            ),
            "median_lifecycle_intervals_sec": median_detailed_phase_intervals(
                manifest["measured"]["metrics"]["lifecycle"] for manifest in manifests
            ),
            "median_gpu_utilization_percent": _median(
                [
                    _metric(manifest, "nvml", "utilization", "average_gpu_percent")
                    for manifest in manifests
                ]
            ),
            "median_power_w": _median(
                [_metric(manifest, "nvml", "power", "average_w") for manifest in manifests]
            ),
            "median_joules_per_frame": _median(
                [_metric(manifest, "nvml", "power", "joules_per_frame") for manifest in manifests]
            ),
            "median_peak_vram_mib": _median(
                [_metric(manifest, "nvml", "memory", "peak_delta_mib") for manifest in manifests]
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


def aggregate_campaign(
    campaign_dir: Path,
    *,
    root: Path,
    idle_seconds: float,
    execution_profile: str = "upstream-default",
    inference_report: Path | None = None,
    preprocessing_report: Path | None = None,
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
        raise CampaignError("Campaign config execution profile does not match aggregation request")
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
    publication_errors = [
        INFERENCE_PARITY_GAP,
        PREPROCESSING_DIAGNOSTIC_GAP,
        PRODUCT_OUTPUT_GAP,
    ]
    if inference_report is not None:
        quality["inference_parity"] = _validate_inference_report(
            inference_report,
            contract=contract,
            root=root,
        )
        publication_errors.remove(INFERENCE_PARITY_GAP)
    if preprocessing_report is not None:
        quality["preprocessing_diagnostic"] = _validate_preprocessing_report(
            preprocessing_report,
            contract=contract,
            root=root,
        )
        publication_errors.remove(PREPROCESSING_DIAGNOSTIC_GAP)
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

    trtvideo_fps = implementation_results["trtvideo"]["statistics"]["median_fps"]
    for result in implementation_results.values():
        median_fps = result["statistics"]["median_fps"]
        result["relative_to_trtvideo_percent"] = (median_fps / trtvideo_fps - 1) * 100

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
        "execution_profile": execution_profile,
        "publishable": publication_ready,
        "publication": {
            "ready": publication_ready,
            "errors": publication_errors,
            "warnings": publication_warnings,
        },
        "workload_id": contract["workload_id"],
        "benchmark_contract_version": contract["benchmark_contract_version"],
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
            "tensor_quality_contract_version": contract["model_space_contract_version"],
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
                "order": [event.implementation for event in events if event.round_index == index],
                "manifests": {
                    name: relative_artifact_path(_manifest_path(campaign_dir, name, index), root)
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
        "--inference-report",
        default=None,
        help="Optional valid shared-input inference report for this exact campaign",
    )
    parser.add_argument(
        "--preprocessing-report",
        default=None,
        help="Optional complete preprocessing diagnostic for this exact campaign",
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
    markdown_path = Path(args.markdown) if args.markdown else campaign_dir / "results.md"
    try:
        summary = aggregate_campaign(
            campaign_dir,
            root=Path(args.root),
            idle_seconds=args.idle_seconds,
            execution_profile=args.execution_profile,
            inference_report=(Path(args.inference_report) if args.inference_report else None),
            preprocessing_report=(
                Path(args.preprocessing_report) if args.preprocessing_report else None
            ),
            product_output_report=(
                Path(args.product_output_report) if args.product_output_report else None
            ),
        )
        write_json(json_path, summary)
        markdown_path.write_text(render_markdown(summary), encoding="utf-8")
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
