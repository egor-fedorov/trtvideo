from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scripts.tuning import workflow
from benchmarks.scripts.tuning.adaptive import CandidatePoint
from benchmarks.scripts.tuning.contract import load_tuning_contract
from benchmarks.scripts.tuning.resource_limit import ResourceLimitEvidence


def test_campaign_resume_starts_untouched_later_stage(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "winner-campaign"

    assert workflow._resume_existing_campaign(campaign_dir, workflow_resume=True) is False


def test_campaign_resume_continues_existing_campaign(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "winner-campaign"
    campaign_dir.mkdir()
    (campaign_dir / "campaign.config.json").write_text("{}\n", encoding="utf-8")

    assert workflow._resume_existing_campaign(campaign_dir, workflow_resume=True) is True


def test_campaign_resume_rejects_partial_directory_without_config(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "winner-campaign"
    campaign_dir.mkdir()

    with pytest.raises(workflow.TuningWorkflowError, match="without its immutable config"):
        workflow._resume_existing_campaign(campaign_dir, workflow_resume=True)


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


def _invalid_product_report(path: Path, *, hashes: tuple[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "implementation": implementation,
                        "status": "invalid",
                        "output_sha256": output_hash,
                        "errors": ["PSNR below threshold"],
                    }
                    for implementation, output_hash in zip(
                        ("vs-mlrt", "TheAnimeScripter"),
                        hashes,
                        strict=True,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )


def test_identical_invalid_product_outputs_are_a_common_path_failure(
    tmp_path: Path,
) -> None:
    report = tmp_path / "product.json"
    _invalid_product_report(report, hashes=("a" * 64, "a" * 64))

    failure = workflow._common_product_output_failure(report)

    assert failure is not None
    assert "same invalid MP4" in failure
    assert "not evidence against either scheduling candidate" in failure


def test_distinct_invalid_product_outputs_remain_candidate_specific(
    tmp_path: Path,
) -> None:
    report = tmp_path / "product.json"
    _invalid_product_report(report, hashes=("a" * 64, "b" * 64))

    assert workflow._common_product_output_failure(report) is None


def _paths(tmp_path: Path) -> workflow.WorkflowPaths:
    return workflow.WorkflowPaths.resolve(
        root=tmp_path,
        benchmarks_dir=Path("benchmarks"),
        manifest=Path("manifest.json"),
        sweep_dir=Path("sweep"),
    )


def _preflight(contract):
    return {
        candidate.candidate_id: {"status": "valid"}
        for candidate in contract.for_implementation("tas")
    }


def test_tas_search_exhausts_grid_then_confirms_best_three(tmp_path, monkeypatch) -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    paths = _paths(tmp_path)
    calls = []
    speeds = dict(
        zip(
            [candidate.candidate_id for candidate in contract.for_implementation("tas")],
            [100, 90, 80, 70],
            strict=True,
        )
    )

    def measure(*, candidate, stage, policy, **_):
        calls.append((candidate.candidate_id, stage, policy.initial_runs))
        return CandidatePoint(candidate, speeds[candidate.candidate_id], 0, "suite.json", 1000, 1)

    monkeypatch.setattr(workflow, "_measure_search_candidate", measure)
    points, limits = workflow._run_tas_reconnaissance(
        contract=contract,
        preflight=_preflight(contract),
        base={},
        paths=paths,
        runner=workflow.MakeRunner(paths),
        resume=False,
    )
    confirmed, graph = workflow._run_confirmation(
        implementation="tas",
        reconnaissance=points,
        contract=contract,
        base={},
        paths=paths,
        runner=workflow.MakeRunner(paths),
        resume=False,
    )
    assert not limits
    assert graph is None
    assert len(points) == 4
    assert [point.candidate.candidate_id for point in confirmed] == list(speeds)[:3]
    assert [count for _, stage, count in calls if stage == "reconnaissance"] == [1] * 4
    assert [count for _, stage, count in calls if stage == "confirmation"] == [3] * 3


def test_tas_oom_excludes_only_affected_io_configuration(tmp_path, monkeypatch) -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    paths = _paths(tmp_path)
    calls = []

    def measure(*, candidate, **_):
        calls.append(candidate.candidate_id)
        if candidate.writer == "ffmpeg":
            return ResourceLimitEvidence("cuda-out-of-memory", "suite", "run", "log", "a" * 64)
        return CandidatePoint(candidate, 100, 0, "suite.json", 1000, 1)

    monkeypatch.setattr(workflow, "_measure_search_candidate", measure)
    points, limits = workflow._run_tas_reconnaissance(
        contract=contract,
        preflight=_preflight(contract),
        base={},
        paths=paths,
        runner=workflow.MakeRunner(paths),
        resume=False,
    )
    assert len(calls) == 4
    assert len(points) == 2
    assert [entry["candidate_id"] for entry in limits] == ["tas-cpu-ffmpeg", "tas-nvdec-ffmpeg"]
    assert (
        limits[0]["execution_profile"] == contract.candidate("tas-cpu-ffmpeg").execution_profile()
    )


def test_tas_thermal_failure_aborts_instead_of_skipping_candidate(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))

    def measure(**_):
        raise workflow.TuningWorkflowError("Invalid throttle reasons observed: sw_thermal_slowdown")

    monkeypatch.setattr(workflow, "_measure_search_candidate", measure)
    with pytest.raises(workflow.TuningWorkflowError, match="sw_thermal_slowdown"):
        workflow._run_tas_reconnaissance(
            contract=contract,
            preflight=_preflight(contract),
            base={},
            paths=paths,
            runner=workflow.MakeRunner(paths),
            resume=True,
        )


def test_tas_make_arguments_and_cli_engine_are_native(tmp_path) -> None:
    paths = _paths(tmp_path)
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))
    base = workflow._base_make_variables(
        paths,
        variant="720p",
        engine=tmp_path / "model.engine",
        tas_engine=tmp_path / "tas.engine",
        gpu_id=0,
    )
    variables = workflow._candidate_variables(contract.candidate("tas-nvdec-nelux"), base)
    assert variables == {
        "MANIFEST": "manifest.json",
        "VARIANT": "720p",
        "ENGINE": "model.engine",
        "TAS_ENGINE": "tas.engine",
        "GPU_ID": "0",
        "EXECUTION_PROFILE": "tuned",
        "TAS_ARGS": "--decode-method nvdec --writer nelux",
    }
    args = workflow.build_parser().parse_args(
        [
            "sweep",
            "--manifest",
            "manifest.json",
            "--variant",
            "720p",
            "--engine",
            "model.engine",
            "--tas-engine",
            "tas.engine",
            "--sweep-dir",
            "sweep",
        ]
    )
    assert args.tas_engine == "tas.engine"


def test_legacy_resume_rejected_before_any_work(tmp_path) -> None:
    path = tmp_path / "search-state.json"
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(workflow.TuningWorkflowError, match="read-only"):
        workflow._verify_resume_schema(tmp_path)


def test_sweep_starts_with_only_preflight_then_rejects_changed_resume(
    tmp_path, monkeypatch
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(Path("benchmarks/tuning/candidates.json").read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(Path("benchmarks/workloads/realesrgan_x2plus_madrid.json").read_bytes())
    for name in ("engine", "tas.engine"):
        (tmp_path / name).write_bytes(b"engine")
    preflight = tmp_path / "sweep/tas-preflight/preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text("{}")
    contract = load_tuning_contract(contract_path)
    monkeypatch.setattr(workflow, "load_tas_preflight", lambda *a, **k: _preflight(contract))
    monkeypatch.setattr(
        workflow, "_run_reconnaissance", lambda **k: ([], "range-exhausted", None, None)
    )
    monkeypatch.setattr(workflow, "_run_tas_reconnaissance", lambda **k: ([], []))
    monkeypatch.setattr(workflow, "_run_confirmation", lambda **k: ([], None))
    monkeypatch.setattr(
        workflow,
        "_write_selection",
        lambda **k: {
            "status": "valid",
            "winners": {
                "vstrt": {"candidate_id": "vstrt-s1-g0"},
                "tas": {"candidate_id": "tas-cpu-ffmpeg"},
            },
        },
    )
    args = workflow.build_parser().parse_args(
        [
            "sweep",
            "--root",
            str(tmp_path),
            "--contract",
            "contract.json",
            "--manifest",
            "manifest.json",
            "--variant",
            "720p",
            "--engine",
            "engine",
            "--tas-engine",
            "tas.engine",
            "--sweep-dir",
            "sweep",
        ]
    )
    assert workflow.run_sweep(args)["status"] == "valid"
    state = json.loads((tmp_path / "sweep/search-state.json").read_text())
    assert state["schema_version"] == 3
    assert state["implementations"]["tas"] == {
        "completion_reason": "grid-exhausted",
        "resource_limits": [],
        "quality_exclusions": [],
        "reconnaissance": [],
        "shortlist": [],
        "confirmation": [],
    }
    args.resume = True
    preflight.write_text("{}\n")
    with pytest.raises(workflow.TuningWorkflowError, match="preflight changed"):
        workflow.run_sweep(args)
