from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.scripts.tuning.contract import load_tuning_contract
from benchmarks.scripts.tuning.rank import (
    PRODUCTS,
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
    candidate: Any,
    *,
    workload: dict[str, Any],
    median_fps: float,
    model_space_valid: bool = True,
) -> None:
    candidate_root = candidate_directory(sweep_dir, candidate)
    run_path = candidate_root / "performance" / "run-01" / "manifest.json"
    run_relative = run_path.relative_to(root).as_posix()
    engine_sha = SHA[f"{candidate.implementation}_engine"]
    image_id = f"sha256:{candidate.implementation}-image"
    profile = candidate.execution_profile()
    encoder = {
        "codec": "h264",
        "rate_control": "cbr",
        "target_bitrate_bps": 60_000_000,
    }
    _write_json(
        run_path,
        {
            "status": "valid",
            "product": PRODUCTS[candidate.implementation],
            "workload_id": workload["id"],
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "variant": "1080p",
            "parameters": {
                **profile,
                "frames": workload["benchmark"]["measured_frames"],
                "warmup_frames": workload["benchmark"]["warmup_frames"],
                "encoder": encoder,
            },
            "assets": {
                "input": {"sha256": SHA["input"]},
                "onnx": {"sha256": SHA["onnx"]},
                "engine": {"sha256": engine_sha},
                "workload_manifest": {
                    "sha256": SHA["workload_manifest"],
                },
            },
            "environment": {
                "image": {
                    "id": image_id,
                    "repository_revision": "a" * 40,
                    "source_dirty": "0",
                }
            },
            "reproducibility": {"publishable": True},
            "measured": {"validation": {"valid": True}},
        },
    )
    _write_json(
        candidate_root / "performance" / "suite.json",
        {
            "status": "valid",
            "workload_id": workload["id"],
            "benchmark_contract_version": workload["benchmark"]["contract_version"],
            "variant": "1080p",
            "parameters": profile,
            "statistics": {
                "median_fps": median_fps,
                "relative_spread": 0.01,
            },
            "runs": [{"manifest": run_relative}],
        },
    )
    _write_json(
        candidate_root / "model-space-parity.json",
        {
            "document_type": "model-space-parity",
            "status": "valid" if model_space_valid else "invalid",
            "publishable": model_space_valid,
            "workload_id": workload["id"],
            "variant": "1080p",
            "assets": {
                "input_sha256": SHA["input"],
                "onnx_sha256": SHA["onnx"],
            },
            "comparisons": [
                {
                    "implementation": PRODUCTS[candidate.implementation],
                    "execution_profile": profile,
                    "status": "valid" if model_space_valid else "invalid",
                    "engine_sha256": engine_sha,
                    "image": {
                        "id": image_id,
                        "repository_revision": "a" * 40,
                        "source_dirty": "0",
                    },
                }
            ],
        },
    )


def _complete_sweep(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    workload = load_manifest(Path("benchmarks/workloads/realesrgan_x2plus_sintel.json"))
    sweep_dir = tmp_path / "artefacts" / "sweep"
    speeds = {
        "vstrt-s2-g0": 10.0,
        "vstrt-s3-g0": 12.0,
        "vstrt-s4-g0": 11.0,
        "vsgan-s2-tauto-g0": 8.0,
        "vsgan-s3-tauto-g0": 8.1,
        "vsgan-s4-tauto-g0": 8.2,
        "vsgan-s5-tauto-g0": 8.3,
        "vsgan-s6-tauto-g0": 8.4,
        "vsgan-s4-g0": 9.0,
        "vsgan-s4-g1": 9.5,
    }
    for candidate in contract.candidates:
        _candidate_evidence(
            tmp_path,
            sweep_dir,
            candidate,
            workload=workload,
            median_fps=speeds[candidate.candidate_id],
        )
    return contract, workload, sweep_dir


def test_rank_selects_fastest_eligible_candidate_per_implementation(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_sweep(tmp_path)

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )

    assert report["status"] == "valid"
    assert report["winners"]["vstrt"]["candidate_id"] == "vstrt-s3-g0"
    assert report["winners"]["vsgan"]["candidate_id"] == "vsgan-s4-g1"


def test_rank_promotes_next_candidate_after_full_quality_failure(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_sweep(tmp_path)
    failed_candidate = contract.candidate("vstrt-s3-g0")
    evidence_path = tmp_path / "artefacts" / "failure.json"
    _write_json(
        evidence_path,
        {
            "document_type": "model-space-parity",
            "status": "invalid",
            "workload_id": workload["id"],
            "variant": "1080p",
            "comparisons": [
                {
                    "implementation": PRODUCTS["vstrt"],
                    "status": "invalid",
                    "execution_profile": failed_candidate.execution_profile(),
                    "errors": ["product output parity failed"],
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
            "vstrt-s3-g0": {
                "reason": "product output parity failed",
                "evidence": evidence_path.relative_to(tmp_path).as_posix(),
            }
        },
    )

    assert report["status"] == "valid"
    assert report["winners"]["vstrt"]["candidate_id"] == "vstrt-s4-g0"
    failed = next(
        candidate
        for candidate in report["candidates"]
        if candidate["candidate_id"] == "vstrt-s3-g0"
    )
    assert failed["status"] == "disqualified"


def test_rank_rejects_incomplete_sweep_even_when_winners_exist(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_sweep(tmp_path)
    missing = (
        candidate_directory(
            sweep_dir,
            contract.candidate("vsgan-s4-g0"),
        )
        / "model-space-parity.json"
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
    assert "vsgan-s4-g0" in report["errors"][0]


def test_rank_excludes_candidate_that_fails_model_space(
    tmp_path: Path,
) -> None:
    contract, workload, sweep_dir = _complete_sweep(tmp_path)
    candidate = contract.candidate("vsgan-s4-g1")
    _candidate_evidence(
        tmp_path,
        sweep_dir,
        candidate,
        workload=workload,
        median_fps=9.5,
        model_space_valid=False,
    )

    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant="1080p",
        sweep_dir=sweep_dir,
        root=tmp_path,
    )

    assert report["status"] == "valid"
    assert report["winners"]["vsgan"]["candidate_id"] == "vsgan-s4-g0"
