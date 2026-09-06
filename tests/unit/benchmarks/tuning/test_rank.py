from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.runtime.suite import compute_suite_statistics
from benchmarks.scripts.tuning import rank
from benchmarks.scripts.tuning.adaptive import CandidatePoint, select_peak_equivalent, shortlist
from benchmarks.scripts.tuning.contract import (
    Candidate,
    MeasurementPolicy,
    TasCandidate,
    TunedCandidate,
    load_tuning_contract,
)
from benchmarks.scripts.tuning.rank import (
    PRODUCTS,
    CandidateAssessment,
    TuningEvidenceError,
    _validate_suite,
    candidate_directory,
    load_tas_preflight,
    rank_tuned_candidates,
)
from benchmarks.scripts.workloads.manifest import load_manifest

SHA = {
    "input": "1" * 64,
    "onnx": "2" * 64,
    "workload_manifest": "3" * 64,
    "vstrt_engine": "4" * 64,
    "tas_engine": "5" * 64,
}


@pytest.fixture(autouse=True)
def isolated_quality_validator(monkeypatch):
    # Quality validates its own raw captures and host assets in separate unit tests.
    monkeypatch.setattr(
        rank, "load_preflight_report", lambda path, **_: json.loads(path.read_text())
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _candidate_evidence(
    root: Path,
    sweep_dir: Path,
    candidate: Candidate,
    *,
    workload: dict[str, Any],
    stage: str,
    policy: MeasurementPolicy,
    median_fps: float,
    relative_spread: float = 0.002,
    peak_vram_mib: float = 4000.0,
    cpu_cores: float = 2.0,
    values_fps: list[float] | None = None,
) -> CandidatePoint:
    candidate_root = candidate_directory(sweep_dir, candidate)
    suite_path = candidate_root / stage / "performance" / "suite.json"
    engine_sha = SHA[f"{candidate.implementation}_engine"]
    image_id = f"sha256:{candidate.implementation}-image"
    profile = candidate.execution_profile()
    encoder = {
        "codec": "h264",
        "rate_control": "cbr",
        "target_bitrate_bps": 60_000_000,
    }
    if values_fps is None:
        values_fps = [median_fps] * policy.initial_runs
        if len(values_fps) > 1:
            values_fps[0] -= median_fps * relative_spread / 2
            values_fps[-1] += median_fps * relative_spread / 2
        if relative_spread > policy.spread_threshold:
            values_fps.extend([median_fps] * policy.extra_runs_on_spread)
    statistics = compute_suite_statistics(values_fps)
    runs = []
    for run_index, fps in enumerate(values_fps, start=1):
        run_path = candidate_root / stage / "performance" / f"run-{run_index:02d}" / "manifest.json"
        _write_json(
            run_path,
            {
                "status": "valid",
                "run_index": run_index,
                "product": PRODUCTS[candidate.implementation],
                "workload_id": workload["id"],
                "benchmark_contract_version": workload["benchmark"]["contract_version"],
                "variant": "1080p",
                "parameters": {
                    **profile,
                    "frames": policy.measured_frames,
                    "warmup_frames": policy.warmup_frames,
                    "bitrate_validation": policy.bitrate_validation,
                    "encoder": encoder,
                },
                "assets": {
                    "input": {"sha256": SHA["input"]},
                    "onnx": {"sha256": SHA["onnx"]},
                    "engine": {"sha256": engine_sha},
                    "workload_manifest": {"sha256": SHA["workload_manifest"]},
                },
                "environment": {
                    "gpu": {
                        "name": "NVIDIA GeForce RTX 3090",
                        "driver_version": "595.84",
                        "power_limit_w": 350.0,
                    },
                    "cpu": {"model": "Test CPU", "logical_cores": 12},
                    "image": {
                        "id": image_id,
                        "repository_revision": "a" * 40,
                        "source_dirty": "0",
                    },
                },
                "reproducibility": {"publishable": True},
                "measured": {
                    "validation": {"valid": True},
                    "metrics": {
                        "end_to_end_fps": fps,
                        "processed_frames": policy.measured_frames,
                        "wall_time_sec": policy.measured_frames / fps,
                        "nvml": {"memory": {"peak_delta_mib": peak_vram_mib}},
                        "cpu": {"average_cores": cpu_cores},
                    },
                },
            },
        )
        runs.append(
            {
                "index": run_index,
                "status": "valid",
                "manifest": run_path.relative_to(root).as_posix(),
                "end_to_end_fps": fps,
            }
        )
    _write_json(
        suite_path,
        {
            "status": "valid",
            "workload_id": workload["id"],
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "variant": "1080p",
            "parameters": {
                **profile,
                "frames": policy.measured_frames,
                "warmup_frames": policy.warmup_frames,
                "initial_runs": policy.initial_runs,
                "extra_runs_on_spread": policy.extra_runs_on_spread,
                "spread_threshold": policy.spread_threshold,
                "max_relative_spread": policy.max_relative_spread,
                "idle_seconds": policy.idle_seconds,
                "bitrate_validation": policy.bitrate_validation,
            },
            "statistics": statistics,
            "runs": runs,
        },
    )
    return CandidatePoint(
        candidate=candidate,
        median_fps=statistics["median_fps"],
        relative_spread=statistics["relative_spread"],
        suite_path=suite_path.relative_to(root).as_posix(),
        median_peak_vram_mib=peak_vram_mib if isinstance(candidate, TasCandidate) else None,
        median_cpu_cores=cpu_cores if isinstance(candidate, TasCandidate) else None,
    )


def _cuda_oom_evidence(
    root: Path,
    sweep_dir: Path,
    candidate: Candidate,
    *,
    workload: dict[str, Any],
    policy: MeasurementPolicy,
) -> dict[str, Any]:
    performance_dir = candidate_directory(sweep_dir, candidate) / "reconnaissance" / "performance"
    run_dir = performance_dir / "run-01"
    stderr_path = run_dir / "warmup.stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(
        "Error Code 2: OutOfMemory (Requested size was 3450470400 bytes.)\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "invalid",
            "artifacts": {
                "warmup_stderr": stderr_path.relative_to(root).as_posix(),
            },
        },
    )
    suite_path = performance_dir / "suite.json"
    _write_json(
        suite_path,
        {
            "status": "invalid",
            "workload_id": workload["id"],
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "variant": "1080p",
            "parameters": {
                **candidate.execution_profile(),
                "frames": policy.measured_frames,
                "warmup_frames": policy.warmup_frames,
                "initial_runs": policy.initial_runs,
                "extra_runs_on_spread": policy.extra_runs_on_spread,
                "spread_threshold": policy.spread_threshold,
                "max_relative_spread": policy.max_relative_spread,
                "idle_seconds": policy.idle_seconds,
                "bitrate_validation": policy.bitrate_validation,
            },
            "runs": [
                {
                    "manifest": manifest_path.relative_to(root).as_posix(),
                }
            ],
        },
    )
    return {
        "candidate_id": candidate.candidate_id,
        **(
            {"num_streams": candidate.num_streams}
            if isinstance(candidate, TunedCandidate)
            else {"execution_profile": candidate.execution_profile()}
        ),
        "kind": "cuda-out-of-memory",
        "suite": suite_path.relative_to(root).as_posix(),
        "run_manifest": manifest_path.relative_to(root).as_posix(),
        "stderr": stderr_path.relative_to(root).as_posix(),
        "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
    }


def _preflight_evidence(root, sweep_dir, contract, workload, *, failures=None) -> Path:
    failures = failures or {}
    entries = []
    for candidate in contract.for_implementation("tas"):
        point = _candidate_evidence(
            root,
            sweep_dir,
            candidate,
            workload=workload,
            stage="quality",
            policy=contract.search.reconnaissance,
            median_fps=1,
        )
        manifest_path = (root / point.suite_path).parent / "run-01/manifest.json"
        quality = {}
        for name, document_type in [
            ("inference_parity", "inference-parity"),
            ("preprocessing_diagnostic", "preprocessing-diagnostic"),
            ("product_output", "product-output-parity"),
        ]:
            comparison = {
                "implementation": "TheAnimeScripter",
                "status": "valid",
                "errors": [],
                "execution_profile": candidate.execution_profile(),
                "engine_sha256": SHA["tas_engine"],
                "run_manifest": manifest_path.relative_to(root).as_posix(),
                "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            if name == "product_output" and candidate.candidate_id in failures:
                comparison.update(status="invalid", errors=[failures[candidate.candidate_id]])
            report = {
                "document_type": document_type,
                "status": "complete"
                if name == "preprocessing_diagnostic"
                else comparison["status"],
                "publishable": comparison["status"] == "valid",
                "acceptance_gate": name == "inference_parity",
                "workload_id": workload["id"],
                "variant": "1080p",
                "reference": {"engine_sha256": SHA["vstrt_engine"]},
                "comparisons": [comparison],
            }
            report_path = sweep_dir / "tas-preflight" / candidate.candidate_id / f"{name}.json"
            _write_json(report_path, report)
            quality[name] = {
                "path": report_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }
        entries.append(
            {
                "candidate_id": candidate.candidate_id,
                "execution_profile": candidate.execution_profile(),
                "status": "disqualified" if candidate.candidate_id in failures else "valid",
                "evidence": {
                    "inference": quality["inference_parity"],
                    "preprocessing": quality["preprocessing_diagnostic"],
                    "product_output": quality["product_output"],
                },
            }
        )
    path = sweep_dir / "tas-preflight/preflight.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "document_type": "tas-quality-preflight",
            "status": "valid",
            "identity": {
                "workload_id": workload["id"],
                "variant": "1080p",
                "files": {
                    "workload_manifest": {"sha256": SHA["workload_manifest"]},
                    "engine": {"sha256": SHA["vstrt_engine"]},
                    "tas_engine": {"sha256": SHA["tas_engine"]},
                },
            },
            "profiles": {entry["candidate_id"]: entry for entry in entries},
        },
    )
    return path


def _complete_search(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    workload = load_manifest(Path("benchmarks/workloads/realesrgan_x2plus_madrid.json"))
    sweep_dir = tmp_path / "artefacts" / "sweep"
    scout_speeds = {
        "vstrt": [10.0, 12.0, 11.9, 11.8, 11.0, 10.8, 10.5, 10.2],
    }
    confirm_speeds = {
        "vstrt": {2: 12.0, 3: 12.05, 4: 11.7},
    }
    states = {}
    for implementation in ("vstrt",):
        reconnaissance = [
            _candidate_evidence(
                tmp_path,
                sweep_dir,
                contract.make_candidate(implementation, streams),
                workload=workload,
                stage="reconnaissance",
                policy=contract.search.reconnaissance,
                median_fps=scout_speeds[implementation][streams - 1],
            )
            for streams in contract.search.stream_range
        ]
        selected = shortlist(reconnaissance, size=contract.search.shortlist_size)
        confirmed = [
            _candidate_evidence(
                tmp_path,
                sweep_dir,
                candidate,
                workload=workload,
                stage="confirmation",
                policy=contract.search.confirmation,
                median_fps=confirm_speeds[implementation][candidate.num_streams],
            )
            for candidate in selected
        ]
        provisional = select_peak_equivalent(
            confirmed,
            equivalence_margin=contract.selection.equivalence_margin,
        )
        assert provisional is not None
        graph_candidate = contract.make_candidate(
            implementation,
            provisional.candidate.num_streams,
            cuda_graph=True,
        )
        graph_speed = 12.1
        confirmed.append(
            _candidate_evidence(
                tmp_path,
                sweep_dir,
                graph_candidate,
                workload=workload,
                stage="confirmation",
                policy=contract.search.confirmation,
                median_fps=graph_speed,
            )
        )
        states[implementation] = {
            "completion_reason": "range-exhausted",
            "early_stop_after_streams": None,
            "resource_limit": None,
            "reconnaissance": [point.as_dict() for point in reconnaissance],
            "shortlist": [candidate.candidate_id for candidate in selected],
            "confirmation": [point.as_dict() for point in confirmed],
            "cuda_graph_probe": graph_candidate.candidate_id,
        }
    tas_points = []
    for candidate, fps, vram in zip(
        contract.for_implementation("tas"),
        [18.0, 19.9, 20.0, 20.1],
        [4000, 5000, 2000, 3000],
        strict=True,
    ):
        tas_points.append(
            _candidate_evidence(
                tmp_path,
                sweep_dir,
                candidate,
                workload=workload,
                stage="reconnaissance",
                policy=contract.search.reconnaissance,
                median_fps=fps,
                relative_spread=0.0,
                peak_vram_mib=vram,
            )
        )
    selected = shortlist(tas_points, size=3)
    tas_confirmation = [
        _candidate_evidence(
            tmp_path,
            sweep_dir,
            point.candidate,
            workload=workload,
            stage="confirmation",
            policy=contract.search.confirmation,
            median_fps=point.median_fps,
            peak_vram_mib=point.median_peak_vram_mib,
        )
        for candidate in selected
        for point in tas_points
        if point.candidate == candidate
    ]
    states["tas"] = {
        "completion_reason": "grid-exhausted",
        "resource_limits": [],
        "quality_exclusions": [],
        "reconnaissance": [point.as_dict() for point in tas_points],
        "shortlist": [candidate.candidate_id for candidate in selected],
        "confirmation": [point.as_dict() for point in tas_confirmation],
    }
    preflight_path = _preflight_evidence(tmp_path, sweep_dir, contract, workload)
    _write_json(
        sweep_dir / "search-state.json",
        {
            "schema_version": 3,
            "document_type": "adaptive-tuning-search",
            "status": "complete",
            "workload_id": workload["id"],
            "variant": "1080p",
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "search_policy": contract.search.as_dict(),
            "selection_policy": contract.selection.as_dict(),
            "tas_preflight": {
                "path": preflight_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
            },
            "implementations": states,
        },
    )
    return contract, workload, sweep_dir


@pytest.fixture(params=["vstrt-s2-g0", "tas-nvdec-ffmpeg"])
def confirmation_evidence(tmp_path, request):
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    workload = load_manifest(Path("benchmarks/workloads/realesrgan_x2plus_madrid.json"))
    candidate = contract.candidate(request.param)
    policy = contract.search.confirmation

    def create(values_fps=None):
        point = _candidate_evidence(
            tmp_path,
            tmp_path / "sweep",
            candidate,
            workload=workload,
            stage="confirmation",
            policy=policy,
            median_fps=100.0,
            values_fps=values_fps,
        )
        suite_path = tmp_path / point.suite_path
        return (
            CandidateAssessment(candidate),
            suite_path,
            {
                "root": tmp_path,
                "suite_path": suite_path,
                "workload_id": workload["id"],
                "variant": "1080p",
                "contract_version": workload["benchmark"]["contract_version"],
                "max_relative_spread": policy.max_relative_spread,
                "policy": policy,
            },
        )

    return create


@pytest.mark.parametrize(
    ("field", "value"),
    [("frames", 300), ("warmup_frames", 24), ("bitrate_validation", False)],
)
def test_confirmation_rejects_raw_run_stage_drift(confirmation_evidence, field, value):
    assessment, suite_path, options = confirmation_evidence()
    for manifest_path in suite_path.parent.glob("run-*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest["parameters"][field] = value
        _write_json(manifest_path, manifest)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert any(f"Performance run changed stage parameters: {field}" in e for e in assessment.errors)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("processed_frames", 300, "processed frame count"),
        ("end_to_end_fps", 500.0, "full-process wall time"),
        ("end_to_end_fps", None, "Performance run FPS"),
        ("wall_time_sec", None, "wall time"),
        ("wall_time_sec", 0, "wall time"),
    ],
)
def test_confirmation_rejects_invalid_raw_metrics(confirmation_evidence, field, value, error):
    assessment, suite_path, options = confirmation_evidence()
    manifest_path = suite_path.parent / "run-01/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["measured"]["metrics"][field] = value
    _write_json(manifest_path, manifest)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert any(error in e for e in assessment.errors)


def test_confirmation_rejects_reused_run_manifest(confirmation_evidence):
    assessment, suite_path, options = confirmation_evidence([100.0] * 3)
    suite = json.loads(suite_path.read_text())
    suite["runs"] = [suite["runs"][0]] * 3
    _write_json(suite_path, suite)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert "Performance suite repeats a run manifest" in assessment.errors


@pytest.mark.parametrize("raw_index", [False, True])
def test_confirmation_rejects_duplicate_run_indices(confirmation_evidence, raw_index):
    assessment, suite_path, options = confirmation_evidence()
    if raw_index:
        path = suite_path.parent / "run-02/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["run_index"] = 1
        _write_json(path, manifest)
    else:
        suite = json.loads(suite_path.read_text())
        suite["runs"][1]["index"] = 1
        _write_json(suite_path, suite)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert any("run index" in e or "run indices" in e for e in assessment.errors)


def test_confirmation_rejects_manifest_from_another_stage(confirmation_evidence):
    assessment, suite_path, options = confirmation_evidence()
    suite = json.loads(suite_path.read_text())
    original_path = suite_path.parent / "run-01/manifest.json"
    foreign_path = (
        suite_path.parent.parent.parent / "reconnaissance/performance/run-01/manifest.json"
    )
    _write_json(foreign_path, json.loads(original_path.read_text()))
    suite["runs"][0]["manifest"] = foreign_path.relative_to(options["root"]).as_posix()
    _write_json(suite_path, suite)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert "Performance run manifest does not belong to this suite" in assessment.errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("median_fps", 200.0),
        ("relative_spread", 0.0),
        ("min_fps", 100.0),
        ("max_fps", 200.0),
        ("values_fps", [100.0] * 3),
    ],
)
def test_confirmation_recomputes_statistics_from_raw_runs(confirmation_evidence, field, value):
    assessment, suite_path, options = confirmation_evidence()
    suite = json.loads(suite_path.read_text())
    suite["statistics"][field] = value
    _write_json(suite_path, suite)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert any(f"statistics differ from raw runs: {field}" in e for e in assessment.errors)
    assert assessment.median_fps == 100.0
    assert assessment.relative_spread == pytest.approx(0.002)


def test_confirmation_rejects_suite_run_fps_changed_from_raw(confirmation_evidence):
    assessment, suite_path, options = confirmation_evidence()
    suite = json.loads(suite_path.read_text())
    suite["runs"][0]["end_to_end_fps"] = 200.0
    _write_json(suite_path, suite)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert "Performance suite run FPS differs from its raw manifest" in assessment.errors


@pytest.mark.parametrize(
    "values",
    [
        [100.0, 100.0, 101.001, 101.001, 101.001],
        [100.0, 100.0, 102.0, 100.0, 100.0],
        [100.0, 100.0, 101.0],
    ],
    ids=["final-spread-drops-below-threshold", "spread-stays-high", "exact-threshold"],
)
def test_confirmation_extension_follows_initial_runs(confirmation_evidence, values):
    assessment, _, options = confirmation_evidence(values)

    _validate_suite(assessment, **options)

    assert assessment.eligible, assessment.errors
    assert assessment.median_fps == compute_suite_statistics(values)["median_fps"]


@pytest.mark.parametrize(
    "values",
    [
        [100.0, 100.0, 101.001],
        [100.0, 100.0, 100.0, 102.0, 102.0],
        [100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0],
    ],
    ids=[
        "missing-extra-runs",
        "late-spread-cannot-justify-extras",
        "unnecessary-extras",
        "partial",
    ],
)
def test_confirmation_rejects_wrong_initial_spread_run_count(confirmation_evidence, values):
    assessment, _, options = confirmation_evidence(values)

    _validate_suite(assessment, **options)

    assert not assessment.eligible
    assert (
        "Performance suite run count does not match its initial spread policy" in assessment.errors
    )


def test_rank_selects_peak_equivalent_resource_efficient_candidate(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )

    assert report["status"] == "valid"
    assert report["winners"]["vstrt"]["candidate_id"] == "vstrt-s2-g0"
    assert report["winners"]["tas"]["candidate_id"] == "tas-nvdec-ffmpeg"
    assert report["winners"]["tas"]["median_peak_vram_mib"] == 2000.0
    assert report["search"]["completion"] == {
        "vstrt": "range-exhausted",
        "tas": "grid-exhausted",
    }


def test_rank_promotes_next_confirmed_candidate_after_quality_failure(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    failed_candidate = contract.candidate("vstrt-s2-g0")
    evidence_path = tmp_path / "artefacts" / "failure.json"
    _write_json(
        evidence_path,
        {
            "document_type": "inference-parity",
            "status": "invalid",
            "workload_id": workload["id"],
            "variant": "1080p",
            "comparisons": [
                {
                    "implementation": PRODUCTS["vstrt"],
                    "status": "invalid",
                    "execution_profile": failed_candidate.execution_profile(),
                    "errors": ["inference parity failed"],
                }
            ],
        },
    )

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
        disqualifications={
            "vstrt-s2-g0": {
                "reason": "inference parity failed",
                "evidence": evidence_path.relative_to(tmp_path).as_posix(),
            }
        },
    )

    assert report["status"] == "valid"
    assert report["winners"]["vstrt"]["candidate_id"] == "vstrt-s2-g1"


def test_rank_rejects_missing_confirmation_evidence(tmp_path: Path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    missing = (
        candidate_directory(
            sweep_dir,
            contract.candidate("tas-nvdec-ffmpeg"),
        )
        / "confirmation"
        / "performance"
        / "suite.json"
    )
    missing.unlink()

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )

    assert report["status"] == "invalid"
    assert "tas-nvdec-ffmpeg" in report["errors"][0]


def test_rank_rejects_environment_drift_between_search_stages(tmp_path: Path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    candidate_dir = candidate_directory(sweep_dir, contract.candidate("vstrt-s2-g0"))
    for manifest_path in (candidate_dir / "confirmation" / "performance").glob(
        "run-*/manifest.json"
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["environment"]["gpu"]["power_limit_w"] = 320.0
        _write_json(manifest_path, manifest)

    with pytest.raises(TuningEvidenceError, match="CPU/GPU environment contract"):
        rank_tuned_candidates(
            contract=contract,
            workload=workload,
            variant="1080p",
            sweep_dir=sweep_dir,
            root=tmp_path,
        )


def test_rank_rejects_unproven_early_stop(tmp_path: Path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    state_path = sweep_dir / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    vstrt = state["implementations"]["vstrt"]
    vstrt["completion_reason"] = "decline-confirmed"
    vstrt["early_stop_after_streams"] = 4
    vstrt["reconnaissance"] = [
        point for point in vstrt["reconnaissance"] if point["num_streams"] in {1, 2, 3, 4, 8}
    ]
    _write_json(state_path, state)

    with pytest.raises(
        TuningEvidenceError,
        match="stopped without a confirmed decline",
    ):
        rank_tuned_candidates(
            contract=contract,
            workload=workload,
            variant="1080p",
            sweep_dir=sweep_dir,
            root=tmp_path,
        )


def test_rank_accepts_hashed_cuda_oom_as_resource_ceiling(tmp_path: Path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    state_path = sweep_dir / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidate = contract.make_candidate("vstrt", 8)
    vstrt = state["implementations"]["vstrt"]
    vstrt["completion_reason"] = "resource-ceiling"
    vstrt["reconnaissance"] = [
        point
        for point in vstrt["reconnaissance"]
        if point["candidate_id"] != candidate.candidate_id
    ]
    vstrt["resource_limit"] = _cuda_oom_evidence(
        tmp_path,
        sweep_dir,
        candidate,
        workload=workload,
        policy=contract.search.reconnaissance,
    )
    _write_json(state_path, state)

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )

    assert report["status"] == "valid"
    assert report["search"]["completion"]["vstrt"] == "resource-ceiling"
    assert report["search"]["resource_limits"]["vstrt"] == vstrt["resource_limit"]


def test_rank_tas_oom_is_per_configuration_not_a_search_stop(tmp_path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    state_path = sweep_dir / "search-state.json"
    state = json.loads(state_path.read_text())
    tas = state["implementations"]["tas"]
    candidate = contract.candidate("tas-cpu-ffmpeg")
    tas["reconnaissance"] = [
        p for p in tas["reconnaissance"] if p["candidate_id"] != candidate.candidate_id
    ]
    tas["resource_limits"] = [
        _cuda_oom_evidence(
            tmp_path,
            sweep_dir,
            candidate,
            workload=workload,
            policy=contract.search.reconnaissance,
        )
    ]
    _write_json(state_path, state)

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )
    assert report["status"] == "valid"
    assert report["search"]["completion"]["tas"] == "grid-exhausted"
    assert report["search"]["resource_limits"]["tas"] == tas["resource_limits"]


@pytest.mark.parametrize("field", ["median_peak_vram_mib", "median_cpu_cores"])
def test_rank_rejects_forged_resource_medians(tmp_path, field) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    path = sweep_dir / "search-state.json"
    state = json.loads(path.read_text())
    state["implementations"]["tas"]["confirmation"][0][field] = 0
    _write_json(path, state)
    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )
    assert report["status"] == "invalid"
    assert any("resource medians" in error for error in report["errors"])


def test_rank_rejects_finite_grid_missing_a_point(tmp_path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    path = sweep_dir / "search-state.json"
    state = json.loads(path.read_text())
    state["implementations"]["tas"]["reconnaissance"].pop(0)
    _write_json(path, state)
    with pytest.raises(TuningEvidenceError, match="incomplete or duplicate grid"):
        rank_tuned_candidates(
            contract=contract,
            workload=workload,
            variant="1080p",
            sweep_dir=sweep_dir,
            root=tmp_path,
        )


def test_rank_rejects_changed_preflight_before_ranking(tmp_path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    preflight = sweep_dir / "tas-preflight/preflight.json"
    preflight.write_text(preflight.read_text() + "\n")
    with pytest.raises(TuningEvidenceError, match="SHA256 changed"):
        rank_tuned_candidates(
            contract=contract,
            workload=workload,
            variant="1080p",
            sweep_dir=sweep_dir,
            root=tmp_path,
        )


def test_rank_excludes_only_quality_failed_configuration_with_evidence(tmp_path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    preflight_path = _preflight_evidence(
        tmp_path,
        sweep_dir,
        contract,
        workload,
        failures={"tas-cpu-ffmpeg": "PSNR below threshold"},
    )
    path = sweep_dir / "search-state.json"
    state = json.loads(path.read_text())
    state["tas_preflight"]["sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    tas = state["implementations"]["tas"]
    tas["quality_exclusions"] = ["tas-cpu-ffmpeg"]
    tas["reconnaissance"] = [
        p for p in tas["reconnaissance"] if p["candidate_id"] != "tas-cpu-ffmpeg"
    ]
    _write_json(path, state)
    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )
    assert report["status"] == "valid"
    assert report["tas_preflight"]["sha256"] == state["tas_preflight"]["sha256"]


@pytest.mark.parametrize(
    "reason", ["decode failed", "Invalid throttle reasons observed: sw_thermal_slowdown"]
)
def test_preflight_infrastructure_failure_cannot_disqualify_candidate(
    tmp_path, reason, monkeypatch
) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    path = _preflight_evidence(
        tmp_path,
        sweep_dir,
        contract,
        workload,
        failures={"tas-cpu-ffmpeg": reason},
    )

    def failed_quality_validator(*args, **kwargs):
        raise rank.PreflightError(reason)

    monkeypatch.setattr(rank, "load_preflight_report", failed_quality_validator)
    with pytest.raises(TuningEvidenceError, match="preflight evidence is invalid"):
        load_tas_preflight(
            path, root=tmp_path, contract=contract, workload=workload, variant="1080p"
        )


def test_preflight_with_no_eligible_tas_stops_search(tmp_path) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    path = _preflight_evidence(
        tmp_path,
        sweep_dir,
        contract,
        workload,
        failures={
            candidate.candidate_id: "SSIM below threshold"
            for candidate in contract.for_implementation("tas")
        },
    )
    with pytest.raises(TuningEvidenceError, match="No TAS configuration"):
        load_tas_preflight(
            path, root=tmp_path, contract=contract, workload=workload, variant="1080p"
        )


@pytest.mark.parametrize("changed", ["engine_sha256", "tas_engine_sha256", "workload_sha256"])
def test_preflight_rejects_stale_engine_or_workload(tmp_path, changed) -> None:
    contract, workload, sweep_dir = _complete_search(tmp_path)
    path = sweep_dir / "tas-preflight/preflight.json"
    expected = {
        "engine_sha256": SHA["vstrt_engine"],
        "tas_engine_sha256": SHA["tas_engine"],
        "workload_sha256": SHA["workload_manifest"],
    }
    expected[changed] = "e" * 64
    with pytest.raises(TuningEvidenceError, match="SHA256 changed"):
        load_tas_preflight(
            path,
            root=tmp_path,
            contract=contract,
            workload=workload,
            variant="1080p",
            **expected,
        )
