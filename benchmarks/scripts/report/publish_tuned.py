"""Build a compact, privacy-reviewed tuned result from raw benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

CANONICAL_ROOT = PurePosixPath("artefacts/benchmarks/comparative/tuning")
DEFAULT_OUTPUT = Path("benchmarks/results/rtx-3090/tuned.json")
DEFAULT_IMPLEMENTATIONS = Path("benchmarks/implementations.json")
DEFAULT_TUNING_CONTRACT = Path("benchmarks/tuning/candidates.json")
ACTIVE_IMPLEMENTATIONS = ("trtvideo", "vstrt", "tas")
EXTERNAL_IMPLEMENTATIONS = ("vstrt", "tas")
CANONICAL_WORKLOADS = (
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "720p"),
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "1080p"),
    ("liveaction_span_madrid", "SPAN", "720p"),
    ("liveaction_span_madrid", "SPAN", "1080p"),
)


class PublicationError(RuntimeError):
    """Raw evidence is incomplete, inconsistent, or not publishable."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"Expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceSource:
    """Resolve copied raw evidence while retaining canonical artifact paths."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def canonical(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.root)
        return str(CANONICAL_ROOT / PurePosixPath(relative.as_posix()))

    def resolve(self, path: str) -> Path:
        try:
            relative = PurePosixPath(path).relative_to(CANONICAL_ROOT)
        except ValueError as exc:
            raise PublicationError(f"Artifact path escapes tuned evidence: {path}") from exc
        resolved = (self.root / Path(*relative.parts)).resolve()
        if not resolved.is_relative_to(self.root):
            raise PublicationError(f"Artifact path escapes tuned evidence: {path}")
        return resolved


def _workloads_for_source(source: EvidenceSource) -> tuple[tuple[str, str, str], ...]:
    """Require the complete canonical Madrid workload contract."""
    missing = [
        f"{base}-matrix.json"
        for base in dict.fromkeys(base for base, _, _ in CANONICAL_WORKLOADS)
        if not (source.root / f"{base}-matrix.json").is_file()
    ]
    if missing:
        raise PublicationError(
            "Tuned evidence does not contain the complete canonical Madrid matrix: "
            + ", ".join(missing)
        )
    return CANONICAL_WORKLOADS


def _verified_artifact(
    source: EvidenceSource, record: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise PublicationError(f"{label} evidence is missing")
    path = source.resolve(record["path"])
    report = _load(path)
    if record.get("sha256") != _digest(path):
        raise PublicationError(f"{label} SHA256 changed or is missing: {path}")
    return path, report


def _check_session_environment(
    actual: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    # These are the same immutable environment fields compared by campaign aggregation.
    for key in ("repository_revision", "gpu", "cpu"):
        if not expected.get(key) or actual.get(key) != expected[key]:
            raise PublicationError(f"{label} session {key} differs or is missing")


def _identity_interpretation(identities: list[dict[str, Any]]) -> str:
    tensors_identical = all(item["candidate_tensor_sha256_sets_identical"] for item in identities)
    outputs_identical = all(item["candidate_mp4_sha256_identical"] for item in identities)
    if tensors_identical and outputs_identical:
        return "Independent external captures are byte-identical for every published workload."
    return (
        "Identity is reported per workload. External implementations use independent "
        "captures and separately built TensorRT engines; "
        "non-identical outputs remain publishable only when the numerical inference "
        "and decoded-product quality gates pass."
    )


def _compact_candidate(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "candidate_id": value["candidate_id"],
        "implementation": value["implementation"],
        "status": value["status"],
        "execution_profile": value["execution_profile"],
        "runner_arguments": value["runner_arguments"],
        "median_fps": value.get("median_fps"),
        "relative_spread": value.get("relative_spread"),
        "errors": value.get("errors", []),
    }
    for key in ("evidence", "median_cpu_cores", "median_peak_vram_mib"):
        if key in value:
            result[key] = value[key]
    return result


def _compact_tas_preflight(source: EvidenceSource, record: dict[str, Any]) -> dict[str, Any]:
    path, report = _verified_artifact(source, record, label="TAS preflight")
    profiles = report.get("profiles")
    expected_profiles = {
        f"tas-{decode}-{writer}": {
            "execution_profile": "tuned",
            "decode_method": decode,
            "writer": writer,
            "cuda_graph": True,
        }
        for decode in ("cpu", "nvdec")
        for writer in ("ffmpeg", "nelux")
    }
    if (
        report.get("schema_version") != 1
        or report.get("document_type") != "tas-quality-preflight"
        or report.get("status") != "valid"
        or not isinstance(profiles, dict)
        or set(profiles) != set(expected_profiles)
    ):
        raise PublicationError(f"TAS preflight is incomplete or has an unsupported schema: {path}")
    candidates = []
    eligible = []
    for candidate_id, expected_profile in expected_profiles.items():
        profile = profiles[candidate_id]
        if (
            not isinstance(profile, dict)
            or profile.get("candidate_id") != candidate_id
            or profile.get("execution_profile") != expected_profile
            or profile.get("status") not in {"valid", "disqualified"}
        ):
            raise PublicationError(f"TAS preflight profile changed: {candidate_id}")
        result_path, result = _verified_artifact(
            source, profile.get("result"), label=f"TAS preflight {candidate_id} result"
        )
        if result_path != path.parent / candidate_id / "result.json" or result != {
            key: value for key, value in profile.items() if key != "result"
        }:
            raise PublicationError(f"TAS preflight profile summary changed: {candidate_id}")
        valid = profile["status"] == "valid"
        if valid:
            eligible.append(candidate_id)
        if profile.get("returncode") != (0 if valid else 2) or bool(profile.get("errors")) == valid:
            raise PublicationError(f"TAS preflight profile outcome changed: {candidate_id}")
        quality = {}
        for name, public_name, status in (
            ("preprocessing", "preprocessing_diagnostic", "complete"),
            ("inference", "inference_parity", "valid"),
            ("product_output", "product_output", "valid"),
        ):
            evidence = profile.get("evidence", {}).get(name)
            if evidence is None and not valid and name == "product_output":
                continue
            _, quality_report = _verified_artifact(
                source, evidence, label=f"TAS preflight {candidate_id} {name}"
            )
            if (valid or name == "preprocessing") and quality_report.get("status") != status:
                raise PublicationError(
                    f"TAS preflight quality status changed: {candidate_id}/{name}"
                )
            quality[public_name] = evidence
        candidates.append(
            {
                "candidate_id": candidate_id,
                "execution_profile": profile["execution_profile"],
                "status": profile["status"],
                "quality": quality,
                "errors": profile["errors"],
                "result": profile["result"],
            }
        )
    if not eligible or report.get("eligible_candidates") != eligible:
        raise PublicationError(f"TAS preflight eligibility changed: {path}")
    return {
        "path": source.canonical(path),
        "sha256": record["sha256"],
        "status": report["status"],
        "eligible_candidates": eligible,
        "candidates": candidates,
    }


def _optional_min(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return min(numeric) if numeric else None


def _optional_max(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return max(numeric) if numeric else None


def _compact_inference(
    source: EvidenceSource,
    report: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    comparisons = []
    for comparison in report["comparisons"]:
        tensors = [item for item in comparison["tensors"] if item["stage"] == "output"]
        comparisons.append(
            {
                "implementation": comparison["implementation"],
                "status": comparison["status"],
                "max_p99_abs": _optional_max([item["metrics"]["p99_abs"] for item in tensors]),
                "max_rmse": _optional_max([item["metrics"]["rmse"] for item in tensors]),
                "min_psnr_db": _optional_min([item["metrics"]["psnr_db"] for item in tensors]),
            }
        )
    return {
        "path": source.canonical(path),
        "sha256": _digest(path),
        "status": report["status"],
        "contract_version": report["contract_version"],
        "frame_indices": report["frame_indices"],
        "thresholds": report["thresholds"],
        "comparisons": comparisons,
    }


def _compact_preprocessing(
    source: EvidenceSource,
    report: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    comparisons = []
    for comparison in report["comparisons"]:
        tensors = comparison["tensors"]
        candidate_statistics = [item["candidate_statistics"] for item in tensors]
        comparisons.append(
            {
                "implementation": comparison["implementation"],
                "status": comparison["status"],
                "max_p99_abs": _optional_max([item["metrics"]["p99_abs"] for item in tensors]),
                "max_rmse": _optional_max([item["metrics"]["rmse"] for item in tensors]),
                "min_psnr_db": _optional_min([item["metrics"]["psnr_db"] for item in tensors]),
                "max_outside_unit_interval_fraction": _optional_max(
                    [item["outside_unit_interval_fraction"] for item in candidate_statistics]
                ),
                "observed_min": _optional_min([item["min"] for item in candidate_statistics]),
                "observed_max": _optional_max([item["max"] for item in candidate_statistics]),
            }
        )
    return {
        "path": source.canonical(path),
        "sha256": _digest(path),
        "status": report["status"],
        "acceptance_gate": False,
        "contract_version": report["contract_version"],
        "frame_indices": report["frame_indices"],
        "comparisons": comparisons,
    }


def _compact_product_output(
    source: EvidenceSource,
    report: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    return {
        "path": source.canonical(path),
        "sha256": _digest(path),
        "status": report["status"],
        "comparisons": [
            {
                "implementation": item["implementation"],
                "status": item["status"],
                "psnr_average_db": item["metrics"]["psnr"]["average_db"],
                "ssim": item["metrics"]["ssim"]["all"],
                "compared_frames": item["metrics"]["psnr"]["frames"],
            }
            for item in report["comparisons"]
        ],
    }


def _run_observations(
    source: EvidenceSource,
    campaign: dict[str, Any],
    implementation: str,
) -> dict[str, Any]:
    manifests = [
        _load(source.resolve(round_value["manifests"][implementation]))
        for round_value in campaign["rounds"]
    ]
    nvml = [manifest["measured"]["metrics"]["nvml"] for manifest in manifests]
    return {
        "peak_temperature_c": max(item["temperature"]["peak_c"] for item in nvml),
        "power_cap_observed": any(item["power"]["power_cap_observed"] for item in nvml),
        "throttle_reasons": sorted(
            {reason for item in nvml for reason in item["throttle_reasons"]}
        ),
    }


def _compact_result(
    source: EvidenceSource,
    campaign: dict[str, Any],
    implementation: str,
    value: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    statistics = value["statistics"]
    run_paths = [source.resolve(item["manifests"][implementation]) for item in campaign["rounds"]]
    runs = [_load(path) for path in run_paths]
    if not runs or any(run.get("status") != "valid" for run in runs):
        raise PublicationError(f"Campaign has missing or invalid {implementation} run evidence")
    for path, run in zip(run_paths, runs, strict=True):
        environment = run["environment"]
        image = environment["image"]
        _check_session_environment(
            {**environment, "repository_revision": image.get("repository_revision")},
            campaign["environment"],
            label=str(path),
        )
        if str(image.get("source_dirty")) != "0":
            raise PublicationError(f"Campaign run was built from dirty source: {path}")
        if implementation in EXTERNAL_IMPLEMENTATIONS:
            recorded = environment.get("implementation", {})
            for key, expected in metadata.items():
                if key != "product" and recorded.get(key) != expected:
                    raise PublicationError(
                        f"{implementation} publication metadata {key} differs from run: {path}"
                    )
    first_run = runs[0]
    runtime = {
        "image": first_run["environment"]["image"],
        "software": first_run["environment"]["software"],
    }
    if implementation == "tas":
        native = first_run["measured"]["validation"].get("runtime_evidence", {})
        if native.get("status") != "valid" or native.get("errors") != []:
            raise PublicationError("TAS result lacks valid native runtime evidence")
        profile = campaign["parameters"]["execution_profiles"][implementation]
        expected = {
            "decode_method": profile["decode_method"],
            "writer": profile["writer"],
            "cuda_graph": True,
            "engine_reused": True,
            "engine_sha256": value["engine_sha256"],
            "onnx_sha256": campaign["assets"]["onnx_sha256"],
        }
        if any(native.get(key) != expected_value for key, expected_value in expected.items()):
            raise PublicationError("TAS native execution differs from the selected campaign")
        if any(run["measured"]["validation"].get("runtime_evidence") != native for run in runs[1:]):
            raise PublicationError("TAS native runtime evidence changed between campaign rounds")
        if native.get("source_revision") != metadata["source_revision"]:
            raise PublicationError("TAS source_revision differs from native runtime evidence")
        for name in ("python", "torch", "tensorrt", "nelux", "ffmpeg"):
            key = f"{name}_version"
            if key in metadata and native.get("runtime", {}).get(name) != metadata[key]:
                raise PublicationError(f"TAS {key} differs from native runtime evidence")
        runtime["software"] = native["runtime"]
        runtime["native_execution"] = {
            key: native[key]
            for key in (
                "source_revision",
                "adapter_sha256",
                "decode_method",
                "writer",
                "cuda_graph",
                "engine_reused",
                "engine_sha256",
                "onnx_sha256",
            )
        }
    return {
        "implementation": implementation,
        "product": value["product"],
        "image_id": value["image_id"],
        "engine_sha256": value["engine_sha256"],
        "relative_to_trtvideo_percent": value["relative_to_trtvideo_percent"],
        "fps_median": statistics["median_fps"],
        "fps_runs": statistics["values_fps"],
        "fps_spread": statistics["relative_spread"],
        "wall_sec": statistics["median_wall_time_sec"],
        "startup_sec": statistics["median_startup_sec"],
        "steady_state_sec": statistics["median_steady_state_frame_loop_sec"],
        "finalize_sec": statistics["median_finalize_mux_sec"],
        "cpu_cores": statistics["median_cpu_cores"],
        "cpu_capacity_percent": statistics["median_cpu_capacity_percent"],
        "gpu_util_percent": statistics["median_gpu_utilization_percent"],
        "power_w": statistics["median_power_w"],
        "joules_per_frame": statistics["median_joules_per_frame"],
        "peak_vram_mib": statistics["median_peak_vram_mib"],
        "output_bitrate_mbps": statistics["median_output_bitrate_mbps"],
        "output_size_mib": statistics["median_output_size_mib"],
        "lifecycle_intervals_sec": statistics["median_lifecycle_intervals_sec"],
        "session_observations": _run_observations(source, campaign, implementation),
        "stability": value["stability"],
        "execution_profile": campaign["parameters"]["execution_profiles"].get(
            implementation, {"execution_profile": campaign["execution_profile"]}
        ),
        "runtime": runtime,
        "engine_manifest_sha256": first_run.get("assets", {})
        .get("engine_manifest", {})
        .get("sha256"),
        "run_evidence": [
            {"path": source.canonical(path), "sha256": _digest(path)} for path in run_paths
        ],
    }


def _intra_session_reproducibility(
    selection: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    comparisons = []
    for implementation in selection["winners"]:
        confirmation_fps = float(selection["winners"][implementation]["median_fps"])
        final_fps = float(campaign["implementations"][implementation]["statistics"]["median_fps"])
        comparisons.append(
            {
                "implementation": implementation,
                "confirmation_fps": confirmation_fps,
                "final_campaign_fps": final_fps,
                "delta_percent": (final_fps / confirmation_fps - 1.0) * 100.0,
            }
        )
    return {
        "scope": "Selected external profiles measured independently within one session",
        "comparisons": comparisons,
        "max_absolute_delta_percent": max(
            abs(float(item["delta_percent"])) for item in comparisons
        ),
    }


def _tensor_set_digest(manifest: dict[str, Any]) -> str:
    records = sorted(
        (str(item["stage"]), int(item["frame_index"]), str(item["sha256"]))
        for item in manifest["artifacts"]
    )
    payload = "".join(f"{stage}\t{index}\t{sha}\n" for stage, index, sha in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def _output_identity(
    source: EvidenceSource,
    base: str,
    workload_name: str,
    variant: str,
    winners: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    winner_key = "__".join(winners[name]["candidate_id"] for name in sorted(winners))
    quality_root = source.root / f"{base}-{variant}" / "winner-quality" / winner_key
    tensor_root = quality_root / "tensor-quality"
    _, product_report = _verified_artifact(
        source, quality["product_output"], label="Product identity report"
    )
    _, inference_report = _verified_artifact(
        source, quality["inference_parity"], label="Inference identity report"
    )
    capture_paths = {
        name: tensor_root / "inference" / name / "manifest.json" for name in sorted(winners)
    }
    captures = {name: _load(path) for name, path in capture_paths.items()}
    for name, capture in captures.items():
        if not any(
            item["implementation"] == capture["implementation"]
            and item.get("capture_manifest_sha256") == _digest(capture_paths[name])
            for item in inference_report["comparisons"]
        ):
            raise PublicationError(f"{name} capture SHA256 differs from inference report")
    tensor_digests = {name: _tensor_set_digest(value) for name, value in captures.items()}
    comparisons = product_report["comparisons"]
    output_digests = {item["implementation"]: item["output_sha256"] for item in comparisons}
    tensor_sets_identical = len(set(tensor_digests.values())) == 1
    mp4_outputs_identical = len(set(output_digests.values())) == 1
    return {
        "workload": workload_name,
        "variant": variant,
        "candidate_tensor_sha256_sets_identical": tensor_sets_identical,
        "candidate_tensor_set_sha256": (
            next(iter(tensor_digests.values())) if tensor_sets_identical else None
        ),
        "candidate_tensor_set_sha256_by_implementation": tensor_digests,
        "candidate_mp4_sha256_identical": mp4_outputs_identical,
        "candidate_mp4_sha256": (
            next(iter(output_digests.values())) if mp4_outputs_identical else None
        ),
        "candidate_mp4_sha256_by_implementation": output_digests,
        "capture_manifest_sha256": {name: _digest(path) for name, path in capture_paths.items()},
        "run_manifest_sha256": {
            item["implementation"]: item["run_manifest_sha256"] for item in comparisons
        },
    }


def _compact_workload(
    source: EvidenceSource,
    base: str,
    workload_name: str,
    variant: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    directory = source.root / f"{base}-{variant}"
    selection_path = directory / "selection.json"
    selection = _load(selection_path)
    matrix = _load(source.root / f"{base}-matrix.json")
    matrix_variant = matrix["variants"][variant]
    campaign_path, campaign = _verified_artifact(
        source, matrix_variant.get("campaign"), label=f"{base}-{variant} campaign"
    )
    if set(campaign.get("implementations", {})) != set(ACTIVE_IMPLEMENTATIONS):
        raise PublicationError("New publications require trtvideo, vstrt, and tas participants")
    if selection.get("status") != "valid" or set(selection.get("winners", {})) != set(
        EXTERNAL_IMPLEMENTATIONS
    ):
        raise PublicationError(
            f"Selection does not contain valid TAS/vs-mlrt winners: {selection_path}"
        )
    if (
        campaign.get("status") != "valid"
        or campaign.get("publishable") is not True
        or campaign.get("execution_profile") != "tuned"
        or campaign.get("variant") != variant
    ):
        raise PublicationError(f"Campaign is not publishable: {campaign_path}")
    quality = {}
    for name, (status, gate) in {
        "inference_parity": ("valid", True),
        "preprocessing_diagnostic": ("complete", False),
        "product_output": ("valid", None),
    }.items():
        path, report = _verified_artifact(
            source, matrix_variant.get("quality", {}).get(name), label=name
        )
        if (
            report.get("status") != status
            or report.get("publishable") is not True
            or report.get("variant") != variant
            or report.get("workload_id") != campaign["workload_id"]
        ):
            raise PublicationError(f"{name} quality evidence is not publishable: {path}")
        if gate is not None:
            if report.get("acceptance_gate") is not gate:
                raise PublicationError(f"{name} acceptance role changed: {path}")
            if report.get("contract_version") != campaign["parameters"].get(
                "tensor_quality_contract_version"
            ):
                raise PublicationError(f"{name} contract version changed: {path}")
        quality[name] = (path, report)
    inference_path, inference_report = quality["inference_parity"]
    preprocessing_path, preprocessing_report = quality["preprocessing_diagnostic"]
    product_path, product_report = quality["product_output"]
    matrix_environment = matrix.get("environment", {})
    campaign_environment = campaign["environment"]
    if matrix_environment.get("repository_revision") != campaign_environment.get(
        "repository_revision"
    ) or any(
        matrix_environment.get("gpu", {}).get(key) is None
        or matrix_environment["gpu"][key] != campaign_environment.get("gpu", {}).get(key)
        for key in ("name", "driver_version", "power_limit_w")
    ):
        raise PublicationError(f"Publication matrix session environment differs: {campaign_path}")
    search_state_path = directory / "search-state.json"
    search = selection["search"]
    return {
        "workload_id": selection["workload_id"],
        "workload": workload_name,
        "variant": variant,
        "benchmark_contract_version": selection["benchmark_contract_version"],
        "status": "valid",
        "selection": {
            "path": source.canonical(selection_path),
            "sha256": _digest(selection_path),
            "policy": selection["selection_policy"],
            "search": {
                "state_path": source.canonical(search_state_path),
                "state_sha256": _digest(search_state_path),
                "completion": search["completion"],
                "resource_limits": search["resource_limits"],
            },
            "winners": selection["winners"],
            "tas_preflight": _compact_tas_preflight(source, selection["tas_preflight"]),
            "reconnaissance": [_compact_candidate(item) for item in selection["reconnaissance"]],
            "candidates": [_compact_candidate(item) for item in selection["candidates"]],
            "disqualifications": selection["disqualifications"],
        },
        "intra_session_reproducibility": _intra_session_reproducibility(selection, campaign),
        "final_campaign": {
            "execution_profile": campaign["execution_profile"],
            "participants": list(ACTIVE_IMPLEMENTATIONS),
            "workload_id": campaign["workload_id"],
            "variant": campaign["variant"],
            "benchmark_contract_version": campaign["benchmark_contract_version"],
            "status": campaign["status"],
            "publishable": campaign["publishable"],
            "campaign": {
                "path": source.canonical(campaign_path),
                "sha256": _digest(campaign_path),
            },
            "environment": campaign["environment"],
            "parameters": campaign["parameters"],
            "assets": campaign["assets"],
            "quality": {
                "inference_parity": _compact_inference(
                    source,
                    inference_report,
                    inference_path,
                ),
                "preprocessing_diagnostic": _compact_preprocessing(
                    source,
                    preprocessing_report,
                    preprocessing_path,
                ),
                "product_output": _compact_product_output(source, product_report, product_path),
            },
            "results": [
                _compact_result(
                    source, campaign, name, campaign["implementations"][name], metadata[name]
                )
                for name in ACTIVE_IMPLEMENTATIONS
            ],
        },
    }


def _implementation_metadata(path: Path) -> dict[str, Any]:
    source = _load(path)["implementations"]
    return {
        "trtvideo": {"product": "trtvideo"},
        "vstrt": {
            key: value
            for key, value in source["vstrt"].items()
            if key in {"source", "source_revision", "version", "tensorrt_version"}
        }
        | {"product": "vs-mlrt"},
        "tas": {
            key: value
            for key, value in source["tas"].items()
            if key
            in {
                "source",
                "source_revision",
                "version",
                "upstream_image",
                "tensorrt_version",
                "exact_model_match",
                "exact_engine_match",
                "execution_profiles",
                "python_version",
                "torch_version",
                "nelux_version",
                "ffmpeg_version",
                "adapter",
                "adapter_sha256",
                "encoder_adapter",
                "engine_builder",
            }
        }
        | {"product": "TheAnimeScripter"},
    }


def build_document(
    source: EvidenceSource,
    implementations_path: Path,
    tuning_contract_path: Path,
) -> dict[str, Any]:
    workload_specs = _workloads_for_source(source)
    metadata = _implementation_metadata(implementations_path)
    workloads = [
        _compact_workload(source, base, workload_name, variant, metadata)
        for base, workload_name, variant in workload_specs
    ]
    first_campaign_path = source.resolve(workloads[0]["final_campaign"]["campaign"]["path"])
    raw_campaign = _load(first_campaign_path)
    first_manifest = source.resolve(raw_campaign["rounds"][0]["manifests"]["trtvideo"])
    manifest = _load(first_manifest)
    environment = manifest["environment"]
    revision = environment["image"]["repository_revision"]
    date_utc = str(manifest["started_at_utc"])[:10]
    for workload in workloads:
        _check_session_environment(
            workload["final_campaign"]["environment"],
            raw_campaign["environment"],
            label=f"{workload['workload_id']} {workload['variant']}",
        )

    matrices = []
    for base in dict.fromkeys(base for base, _, _ in workload_specs):
        path = source.root / f"{base}-matrix.json"
        evidence = _load(path)
        if evidence["status"] != "valid" or evidence["publishable"] is not True:
            raise PublicationError(f"Publication matrix is not publishable: {path}")
        if evidence["environment"]["repository_revision"] != revision:
            raise PublicationError(f"Publication matrix revision differs: {path}")
        matrices.append(
            {"path": source.canonical(path), "sha256": _digest(path), "evidence": evidence}
        )

    identities = [
        _output_identity(
            source,
            base,
            workload_name,
            variant,
            workload["selection"]["winners"],
            workload["final_campaign"]["quality"],
        )
        for (base, workload_name, variant), workload in zip(workload_specs, workloads, strict=True)
    ]
    independent = {
        "capture_manifest_sha256_differ": all(
            len(set(item["capture_manifest_sha256"].values())) == 2 for item in identities
        ),
        "run_manifest_sha256_differ": all(
            len(set(item["run_manifest_sha256"].values())) == 2 for item in identities
        ),
        "container_image_id_differ": all(
            workload["final_campaign"]["results"][1]["image_id"]
            != workload["final_campaign"]["results"][2]["image_id"]
            for workload in workloads
        ),
        "engine_sha256_differ": all(
            workload["final_campaign"]["results"][1]["engine_sha256"]
            != workload["final_campaign"]["results"][2]["engine_sha256"]
            for workload in workloads
        ),
    }
    if not all(value for key, value in independent.items() if key != "engine_sha256_differ"):
        raise PublicationError("External provenance is not independent")

    return {
        "schema_version": 6,
        "document_type": "published_tuned_results",
        "status": "valid",
        "publishable": True,
        "scope": {
            "date_utc": date_utc,
            "measurement_revision": revision,
            "source_dirty": False,
            "execution_profile": "tuned",
            "participants": list(ACTIVE_IMPLEMENTATIONS),
            "claim_scope": (
                "Best validated throughput selected by the predeclared vs-mlrt stream "
                "search and TAS categorical I/O grid with independent confirmation, "
                "followed by independent quality gates and rotated winner campaigns."
            ),
        },
        "environment": {
            "hardware": {"cpu": environment["cpu"], "gpu": environment["gpu"]},
            "project_runtime": {
                "image": environment["image"],
                "software": environment["software"],
            },
            "implementations": metadata,
        },
        "methodology": {
            "measured_frames": 1000,
            "warmup_frames": {"realesrgan_x2plus": 30, "span": 100},
            "initial_rounds": 3,
            "extra_rounds_on_spread": 2,
            "idle_seconds": 10,
            "spread_threshold": 0.05,
            "stability_policy": "full-range-3-then-consensus-4-of-5",
            "selection_uses_separate_winner_campaign": True,
            "quality_gate_uses_selected_winners": True,
            "adaptive_search": _load(tuning_contract_path),
        },
        "external_output_identity": {
            "status": "verified",
            "all_candidate_tensor_sets_identical": all(
                item["candidate_tensor_sha256_sets_identical"] for item in identities
            ),
            "all_candidate_mp4_outputs_identical": all(
                item["candidate_mp4_sha256_identical"] for item in identities
            ),
            "interpretation": _identity_interpretation(identities),
            "tensor_set_digest": (
                "SHA-256 of ordered stage<TAB>frame_index<TAB>artifact_sha256<LF> records"
            ),
            "independent_provenance": independent,
            "workloads": identities,
        },
        "publication_matrices": matrices,
        "workloads": workloads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--implementations", type=Path, default=DEFAULT_IMPLEMENTATIONS)
    parser.add_argument("--tuning-contract", type=Path, default=DEFAULT_TUNING_CONTRACT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document = build_document(
        EvidenceSource(args.source_dir),
        args.implementations,
        args.tuning_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Published tuned results: {args.output}")


if __name__ == "__main__":
    main()
