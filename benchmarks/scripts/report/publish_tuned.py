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
CANONICAL_WORKLOADS = (
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "720p"),
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "1080p"),
    ("liveaction_span_madrid", "SPAN", "720p"),
    ("liveaction_span_madrid", "SPAN", "1080p"),
)
LEGACY_SINTEL_WORKLOADS = (
    ("realesrgan_x2plus_sintel", "RealESRGAN_x2plus", "720p"),
    ("realesrgan_x2plus_sintel", "RealESRGAN_x2plus", "1080p"),
    ("liveaction_span_sintel", "SPAN", "720p"),
    ("liveaction_span_sintel", "SPAN", "1080p"),
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
        return self.root / Path(*relative.parts)


def _workloads_for_source(source: EvidenceSource) -> tuple[tuple[str, str, str], ...]:
    """Select the canonical contract while retaining legacy Sintel publication support."""
    for workloads in (CANONICAL_WORKLOADS, LEGACY_SINTEL_WORKLOADS):
        bases = {base for base, _, _ in workloads}
        if all((source.root / f"{base}-matrix.json").is_file() for base in bases):
            return workloads
    raise PublicationError("Tuned evidence does not contain a complete known workload matrix")


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
    if value.get("evidence"):
        result["evidence"] = value["evidence"]
    return result


def _compact_model_space(
    source: EvidenceSource,
    report: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    comparisons = []
    for comparison in report["comparisons"]:
        tensors = comparison["tensors"]
        psnr_values = [item["metrics"]["psnr_db"] for item in tensors]
        comparisons.append(
            {
                "implementation": comparison["implementation"],
                "status": comparison["status"],
                "max_p99_abs": max(item["metrics"]["p99_abs"] for item in tensors),
                "max_rmse": max(item["metrics"]["rmse"] for item in tensors),
                "min_psnr_db": min(value for value in psnr_values if value is not None),
            }
        )
    return {
        "path": source.canonical(path),
        "sha256": _digest(path),
        "status": report["status"],
        "frame_indices": report["frame_indices"],
        "thresholds": report["thresholds"],
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
) -> dict[str, Any]:
    statistics = value["statistics"]
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
    }


def _intra_session_reproducibility(
    selection: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    comparisons = []
    for implementation in ("vstrt", "vsgan"):
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
) -> dict[str, Any]:
    winner_key = f"{winners['vsgan']['candidate_id']}__{winners['vstrt']['candidate_id']}"
    quality_root = source.root / f"{base}-{variant}" / "winner-quality" / winner_key
    model_root = quality_root / "model-space"
    product_report = _load(quality_root / "product-output" / "product-output-parity.json")
    capture_paths = {name: model_root / name / "manifest.json" for name in ("vstrt", "vsgan")}
    captures = {name: _load(path) for name, path in capture_paths.items()}
    tensor_digests = {name: _tensor_set_digest(value) for name, value in captures.items()}
    comparisons = product_report["comparisons"]
    output_digests = {item["implementation"]: item["output_sha256"] for item in comparisons}
    if len(set(tensor_digests.values())) != 1 or len(set(output_digests.values())) != 1:
        raise PublicationError(f"External outputs differ for {base}/{variant}")
    return {
        "workload": workload_name,
        "variant": variant,
        "candidate_tensor_sha256_sets_identical": True,
        "candidate_tensor_set_sha256": next(iter(tensor_digests.values())),
        "candidate_mp4_sha256_identical": True,
        "candidate_mp4_sha256": next(iter(output_digests.values())),
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
) -> dict[str, Any]:
    directory = source.root / f"{base}-{variant}"
    selection_path = directory / "selection.json"
    selection = _load(selection_path)
    matrix = _load(source.root / f"{base}-matrix.json")
    matrix_variant = matrix["variants"][variant]
    campaign_path = source.resolve(matrix_variant["campaign"]["path"])
    campaign = _load(campaign_path)
    model_path = source.resolve(matrix_variant["quality"]["model_space"]["path"])
    product_path = source.resolve(matrix_variant["quality"]["product_output"]["path"])
    model_report = _load(model_path)
    product_report = _load(product_path)
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
            "reconnaissance": [_compact_candidate(item) for item in selection["reconnaissance"]],
            "candidates": [_compact_candidate(item) for item in selection["candidates"]],
            "disqualifications": selection["disqualifications"],
        },
        "intra_session_reproducibility": _intra_session_reproducibility(selection, campaign),
        "final_campaign": {
            "execution_profile": campaign["execution_profile"],
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
                "model_space": _compact_model_space(source, model_report, model_path),
                "product_output": _compact_product_output(source, product_report, product_path),
            },
            "results": [
                _compact_result(source, campaign, name, campaign["implementations"][name])
                for name in ("trtvideo", "vsgan", "vstrt")
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
        "vsgan": {
            key: value
            for key, value in source["vsgan"].items()
            if key
            in {
                "source",
                "source_revision",
                "version",
                "upstream_image",
                "tensorrt_version",
                "exact_model_match",
                "exact_engine_match",
            }
        }
        | {"product": "VSGAN-tensorrt-docker"},
    }


def build_document(
    source: EvidenceSource,
    implementations_path: Path,
    tuning_contract_path: Path,
) -> dict[str, Any]:
    workload_specs = _workloads_for_source(source)
    workloads = [
        _compact_workload(source, base, workload_name, variant)
        for base, workload_name, variant in workload_specs
    ]
    first_campaign_path = source.resolve(workloads[0]["final_campaign"]["campaign"]["path"])
    raw_campaign = _load(first_campaign_path)
    first_manifest = source.resolve(raw_campaign["rounds"][0]["manifests"]["trtvideo"])
    manifest = _load(first_manifest)
    environment = manifest["environment"]
    revision = environment["image"]["repository_revision"]
    date_utc = str(manifest["started_at_utc"])[:10]

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
        _output_identity(source, base, workload_name, variant, workload["selection"]["winners"])
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
    if not all(independent.values()):
        raise PublicationError("External provenance is not independent")

    return {
        "schema_version": 4,
        "document_type": "published_tuned_results",
        "status": "valid",
        "publishable": True,
        "scope": {
            "date_utc": date_utc,
            "measurement_revision": revision,
            "source_dirty": False,
            "execution_profile": "tuned",
            "claim_scope": (
                "Best validated throughput selected by the predeclared adaptive two-stage "
                "search, followed by independent quality gates and rotated winner campaigns."
            ),
        },
        "environment": {
            "hardware": {"cpu": environment["cpu"], "gpu": environment["gpu"]},
            "project_runtime": {
                "image": environment["image"],
                "software": environment["software"],
            },
            "implementations": _implementation_metadata(implementations_path),
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
            "interpretation": (
                "vs-mlrt and VSGAN execute libvstrt.so through the same VapourSynth graph "
                "and produce byte-identical candidate tensors and MP4 outputs."
            ),
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
