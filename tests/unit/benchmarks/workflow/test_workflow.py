from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.workflow.cli import _require_supported_python
from benchmarks.scripts.workflow.matrix import load_workflow_matrix
from benchmarks.scripts.workflow.orchestrator import (
    Step,
    WorkflowError,
    WorkflowOptions,
    WorkflowState,
    build_plan,
    run_plan,
)

ROOT = Path(__file__).resolve().parents[4]
MATRIX_PATH = ROOT / "benchmarks/workflows/canonical.json"


def _targets(plan: tuple[Step, ...]) -> list[str]:
    return [step.command[3] for step in plan]


def test_canonical_matrix_contains_two_models_and_two_resolutions() -> None:
    matrix = load_workflow_matrix(MATRIX_PATH)

    assert matrix.workload_keys == ("realesrgan", "span")
    assert {
        workload.key: workload.tuning_contract for workload in matrix.workloads
    } == {
        "realesrgan": "benchmarks/tuning/candidates.json",
        "span": "benchmarks/tuning/span_candidates.json",
    }
    assert [selection.key for selection in matrix.select(
        workload_key=None,
        variant_name=None,
    )] == [
        "realesrgan-720p",
        "realesrgan-1080p",
        "span-720p",
        "span-1080p",
    ]


def test_project_goal_covers_complete_selected_matrix_without_competitors() -> None:
    matrix = load_workflow_matrix(MATRIX_PATH)

    plan = build_plan(
        root=ROOT,
        matrix=matrix,
        options=WorkflowOptions(goal="project"),
    )
    targets = _targets(plan)

    assert len(plan) == 18
    assert targets.count("prepare") == 2
    assert targets.count("build-project-engine") == 4
    assert targets.count("run-project") == 8
    smoke_steps = [step for step in plan if step.key.startswith("smoke:")]
    assert all(
        any("--skip-bitrate-validation" in argument for argument in step.command)
        for step in smoke_steps
    )
    assert "build-vsgan" not in targets
    assert "quality-gates" not in targets


def test_comparative_goal_covers_build_quality_and_campaign() -> None:
    matrix = load_workflow_matrix(MATRIX_PATH)

    plan = build_plan(
        root=ROOT,
        matrix=matrix,
        options=WorkflowOptions(
            goal="comparative",
            workload_key="span",
            variant_name="1080p",
            resume=True,
        ),
    )
    targets = _targets(plan)

    assert len(plan) == 13
    assert "build-vstrt" in targets
    assert "build-vsgan" in targets
    assert "build-vsgan-engine" in targets
    assert "quality-gates" in targets
    assert targets[-1] == "run-comparative"
    assert "EXECUTION_PROFILE=upstream-default" in plan[-1].command
    assert "RESUME=1" in plan[-1].command


def test_tuned_goal_runs_each_phase_then_verifies_both_model_matrices() -> None:
    matrix = load_workflow_matrix(MATRIX_PATH)

    plan = build_plan(
        root=ROOT,
        matrix=matrix,
        options=WorkflowOptions(goal="tuned"),
    )
    targets = _targets(plan)

    assert len(plan) == 42
    assert targets.count("run-tuned-sweep") == 4
    assert targets.count("run-tuned-quality") == 4
    assert targets.count("run-tuned-campaign") == 4
    assert targets.count("verify-tuned-matrix") == 2
    first_quality = targets.index("run-tuned-quality")
    last_sweep = len(targets) - 1 - targets[::-1].index("run-tuned-sweep")
    first_campaign = targets.index("run-tuned-campaign")
    assert last_sweep < first_quality < first_campaign
    tuned_steps = [step for step in plan if step.key.startswith("tuned:")]
    for step in tuned_steps:
        if step.key.endswith(":realesrgan") or step.key.endswith(":span"):
            continue
        expected_contract = (
            "benchmarks/tuning/span_candidates.json"
            if ":span-" in step.key
            else "benchmarks/tuning/candidates.json"
        )
        assert f"TUNING_CONTRACT={expected_contract}" in step.command


def test_diagnostics_goal_runs_all_ceilings_and_one_nsight_trace() -> None:
    matrix = load_workflow_matrix(MATRIX_PATH)

    plan = build_plan(
        root=ROOT,
        matrix=matrix,
        options=WorkflowOptions(goal="diagnostics"),
    )
    targets = _targets(plan)

    assert targets.count("run-trtexec") == 4
    assert targets.count("profile-nsight") == 1
    assert "build-vsgan" not in targets


def test_resume_skips_only_successfully_recorded_steps(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    context = {"goal": "project", "repository_revision": "revision"}
    state = WorkflowState.open(state_path, context=context, resume=False)
    plan = (
        Step(key="first", label="First", command=("first",)),
        Step(key="second", label="Second", command=("second",)),
    )
    executed: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _cwd: Path) -> None:
        executed.append(command)

    run_plan(
        plan[:1],
        root=tmp_path,
        state=state,
        dry_run=False,
        executor=execute,
    )
    resumed = WorkflowState.open(state_path, context=context, resume=True)
    run_plan(
        plan,
        root=tmp_path,
        state=resumed,
        dry_run=False,
        executor=execute,
    )

    assert executed == [("first",), ("second",)]
    assert resumed.completed_keys == {"first", "second"}


def test_resume_rejects_different_context(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = WorkflowState.open(
        state_path,
        context={"goal": "project"},
        resume=False,
    )
    state.complete("build")

    with pytest.raises(WorkflowError, match="does not match"):
        WorkflowState.open(
            state_path,
            context={"goal": "comparative"},
            resume=True,
        )


def test_launcher_dry_run_works_without_gpu() -> None:
    result = subprocess.run(
        (
            str(ROOT / "benchmarks/bin/run-benchmark.sh"),
            "comparative",
            "--workload",
            "span",
            "--variant",
            "1080p",
            "--dry-run",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "matrix: span-1080p" in result.stdout
    assert "quality-gates" in result.stdout
    assert "run-comparative" in result.stdout


def test_workflow_rejects_unsupported_host_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 13))

    with pytest.raises(WorkflowError, match=r">=3\.10,<3\.13"):
        _require_supported_python()
