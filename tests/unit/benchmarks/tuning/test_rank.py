from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.tuning.adaptive import CandidatePoint, select_peak_equivalent, shortlist
from benchmarks.scripts.tuning.contract import (
    MeasurementPolicy,
    TunedCandidate,
    load_tuning_contract,
)
from benchmarks.scripts.tuning.rank import (
    PRODUCTS,
    TuningEvidenceError,
    candidate_directory,
    rank_tuned_candidates,
)
from benchmarks.scripts.workloads.manifest import load_manifest

SHA = {
    "input": "1" * 64,
    "onnx": "2" * 64,
    "workload_manifest": "3" * 64,
    "vstrt_engine": "4" * 64,
    "vsgan_engine": "5" * 64,
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _candidate_evidence(
    root: Path,
    sweep_dir: Path,
    candidate: TunedCandidate,
    *,
    workload: dict[str, Any],
    stage: str,
    policy: MeasurementPolicy,
    median_fps: float,
    relative_spread: float = 0.002,
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
    run_count = policy.initial_runs
    if relative_spread > policy.spread_threshold:
        run_count += policy.extra_runs_on_spread
    runs = []
    for run_index in range(1, run_count + 1):
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
                "measured": {"validation": {"valid": True}},
            },
        )
        runs.append(
            {
                "manifest": run_path.relative_to(root).as_posix(),
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
            "statistics": {
                "median_fps": median_fps,
                "relative_spread": relative_spread,
            },
            "runs": runs,
        },
    )
    return CandidatePoint(
        candidate=candidate,
        median_fps=median_fps,
        relative_spread=relative_spread,
        suite_path=suite_path.relative_to(root).as_posix(),
    )


def _cuda_oom_evidence(
    root: Path,
    sweep_dir: Path,
    candidate: TunedCandidate,
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
        "num_streams": candidate.num_streams,
        "kind": "cuda-out-of-memory",
        "suite": suite_path.relative_to(root).as_posix(),
        "run_manifest": manifest_path.relative_to(root).as_posix(),
        "stderr": stderr_path.relative_to(root).as_posix(),
        "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
    }


def _complete_search(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    workload = load_manifest(Path("benchmarks/workloads/realesrgan_x2plus_madrid.json"))
    sweep_dir = tmp_path / "artefacts" / "sweep"
    scout_speeds = {
        "vstrt": [10.0, 12.0, 11.9, 11.8, 11.0, 10.8, 10.5, 10.2],
        "vsgan": [12.0, 15.0, 18.0, 19.8, 20.0, 20.1, 19.5, 19.0],
    }
    confirm_speeds = {
        "vstrt": {2: 12.0, 3: 12.05, 4: 11.7},
        "vsgan": {4: 19.8, 5: 20.0, 6: 20.1},
    }
    states = {}
    for implementation in PRODUCTS:
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
        graph_speed = 12.1 if implementation == "vstrt" else 20.25
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
    _write_json(
        sweep_dir / "search-state.json",
        {
            "schema_version": 2,
            "document_type": "adaptive-tuning-search",
            "status": "complete",
            "workload_id": workload["id"],
            "variant": "1080p",
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "search_policy": contract.search.as_dict(),
            "selection_policy": contract.selection.as_dict(),
            "implementations": states,
        },
    )
    return contract, workload, sweep_dir


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
    assert report["winners"]["vsgan"]["candidate_id"] == "vsgan-s5-tauto-g1"
    assert report["search"]["completion"] == {
        "vstrt": "range-exhausted",
        "vsgan": "range-exhausted",
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
            contract.candidate("vsgan-s5-tauto-g1"),
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
    assert "vsgan-s5-tauto-g1" in report["errors"][0]


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
