from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scripts.tuning import workflow
from benchmarks.scripts.tuning.adaptive import CandidatePoint
from benchmarks.scripts.tuning.contract import load_tuning_contract
from benchmarks.scripts.tuning.resource_limit import ResourceLimitEvidence


def test_reconnaissance_stops_at_hashed_resource_ceiling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    paths = workflow.WorkflowPaths.resolve(
        root=tmp_path,
        benchmarks_dir=tmp_path / "benchmarks",
        manifest=tmp_path / "manifest.json",
        sweep_dir=tmp_path / "sweep",
    )

    def measure(*, candidate, **_):
        if candidate.num_streams == 8:
            return ResourceLimitEvidence(
                kind="cuda-out-of-memory",
                suite_path="suite.json",
                run_manifest_path="run-01/manifest.json",
                stderr_path="run-01/warmup.stderr.log",
                stderr_sha256="a" * 64,
            )
        return CandidatePoint(
            candidate=candidate,
            median_fps=10.0,
            relative_spread=0.0,
            suite_path=f"s{candidate.num_streams}/suite.json",
        )

    monkeypatch.setattr(workflow, "_measure_search_candidate", measure)

    points, reason, early_stop, resource_limit = workflow._run_reconnaissance(
        implementation="vstrt",
        contract=contract,
        base={},
        paths=paths,
        runner=workflow.MakeRunner(paths),
        resume=False,
    )

    assert [point.candidate.num_streams for point in points] == list(range(1, 8))
    assert reason == "resource-ceiling"
    assert early_stop is None
    assert resource_limit == {
        "candidate_id": "vstrt-s8-g0",
        "num_streams": 8,
        "kind": "cuda-out-of-memory",
        "suite": "suite.json",
        "run_manifest": "run-01/manifest.json",
        "stderr": "run-01/warmup.stderr.log",
        "stderr_sha256": "a" * 64,
    }


def _invalid_inference_report(path: Path, error: str) -> None:
    path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "implementation": "vs-mlrt",
                        "status": "invalid",
                        "errors": [error],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_inference_output_failure_can_disqualify_candidate(tmp_path: Path) -> None:
    report = tmp_path / "inference.json"
    _invalid_inference_report(report, "output frame 499: rmse exceeded")

    assert workflow._failed_inference_implementations(report) == {
        "vstrt": "output frame 499: rmse exceeded"
    }


def test_shared_input_failure_aborts_tuning_instead_of_disqualifying_candidates(
    tmp_path: Path,
) -> None:
    report = tmp_path / "inference.json"
    _invalid_inference_report(report, "input frame 499: canonical input tensor differs")

    with pytest.raises(workflow.TuningWorkflowError, match="Shared-input inference"):
        workflow._failed_inference_implementations(report)
