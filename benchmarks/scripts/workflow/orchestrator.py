"""Build and execute complete benchmark workflows from low-level Make targets."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.scripts.workflow.matrix import Selection, WorkflowMatrix

GOALS = ("project", "comparative", "tuned", "diagnostics")
COMPARISON_MODES = ("parity", "upstream-default")
SMOKE_ARGS = (
    "--frames 120 --warmup-frames 24 --runs 1 "
    "--extra-runs 0 --idle-seconds 0 --skip-bitrate-validation"
)


class WorkflowError(RuntimeError):
    """Raised when a complete workflow cannot be planned or resumed safely."""


@dataclass(frozen=True)
class WorkflowOptions:
    """User-selected workflow scope."""

    goal: str
    mode: str = "parity"
    workload_key: str | None = None
    variant_name: str | None = None
    gpu_id: int = 0
    resume: bool = False

    def validate(self) -> None:
        if self.goal not in GOALS:
            raise WorkflowError(f"Unknown workflow goal: {self.goal}")
        if self.mode not in COMPARISON_MODES:
            raise WorkflowError(f"Unknown comparison mode: {self.mode}")
        if self.goal != "comparative" and self.mode != "parity":
            raise WorkflowError("--mode applies only to the comparative goal")
        if self.gpu_id < 0:
            raise WorkflowError("GPU id must be non-negative")


@dataclass(frozen=True)
class Step:
    """One resumable low-level operation."""

    key: str
    label: str
    command: tuple[str, ...]


def _make(
    directory: Path,
    target: str,
    *variables: str,
) -> tuple[str, ...]:
    return ("make", "-C", str(directory), target, *variables)


def _selection_variables(selection: Selection, gpu_id: int) -> tuple[str, ...]:
    return (
        f"MANIFEST={selection.workload.manifest}",
        f"VARIANT={selection.variant.name}",
        f"ENGINE={selection.variant.engine}",
        f"VSGAN_ENGINE={selection.variant.vsgan_engine}",
        f"ONNX={selection.variant.onnx}",
        f"GPU_ID={gpu_id}",
    )


def _build_steps(
    *,
    root: Path,
    options: WorkflowOptions,
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = [
        Step(
            key="build:production",
            label="Build production image",
            command=_make(root, "build"),
        ),
        Step(
            key="build:benchmark",
            label="Build project benchmark image",
            command=_make(benchmark_dir, "build"),
        ),
    ]
    if options.goal in {"comparative", "tuned"}:
        steps.extend(
            (
                Step(
                    key="build:vstrt",
                    label="Build vstrt benchmark image",
                    command=_make(benchmark_dir, "build-vstrt"),
                ),
                Step(
                    key="build:vsgan",
                    label="Build pinned VSGAN benchmark image",
                    command=_make(benchmark_dir, "build-vsgan"),
                ),
            )
        )
    return steps


def _asset_steps(
    *,
    root: Path,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps: list[Step] = []
    seen: set[str] = set()
    for selection in selections:
        workload = selection.workload
        if workload.key in seen:
            continue
        seen.add(workload.key)
        variable = f"MANIFEST={workload.manifest}"
        steps.extend(
            (
                Step(
                    key=f"assets:{workload.key}:prepare",
                    label=f"Prepare {workload.key} assets",
                    command=_make(benchmark_dir, "prepare", variable),
                ),
                Step(
                    key=f"assets:{workload.key}:verify",
                    label=f"Verify {workload.key} assets",
                    command=_make(benchmark_dir, "verify", variable),
                ),
            )
        )
    return steps


def _engine_steps(
    *,
    root: Path,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = []
    for selection in selections:
        variables = _selection_variables(selection, options.gpu_id)
        steps.append(
            Step(
                key=f"engine:{selection.key}:project",
                label=f"Build {selection.key} TRT11 engine",
                command=_make(
                    benchmark_dir,
                    "build-project-engine",
                    *variables,
                ),
            )
        )
        if options.goal in {"comparative", "tuned"}:
            steps.append(
                Step(
                    key=f"engine:{selection.key}:vsgan",
                    label=f"Build {selection.key} VSGAN TRT10 engine",
                    command=_make(
                        benchmark_dir,
                        "build-vsgan-engine",
                        *variables,
                    ),
                )
            )
    return steps


def _smoke_steps(
    *,
    root: Path,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = []
    smoke_mode = options.mode if options.goal == "comparative" else "parity"
    for selection in selections:
        variables = _selection_variables(selection, options.gpu_id)
        output_root = (
            "artefacts/benchmarks/workflows/smoke/"
            f"{options.goal}/{selection.key}"
        )
        steps.append(
            Step(
                key=f"smoke:{selection.key}:project",
                label=f"Smoke-test project on {selection.key}",
                command=_make(
                    benchmark_dir,
                    "run-project",
                    *variables,
                    f"PROJECT_OUTPUT_DIR={output_root}/ai-media",
                    f"ARGS={SMOKE_ARGS}",
                ),
            )
        )
        if options.goal in {"comparative", "tuned"}:
            common = (
                *variables,
                f"VAPOURSYNTH_MODE={smoke_mode}",
                f"ARGS={SMOKE_ARGS}",
            )
            steps.extend(
                (
                    Step(
                        key=f"smoke:{selection.key}:vstrt",
                        label=f"Smoke-test vstrt on {selection.key}",
                        command=_make(
                            benchmark_dir,
                            "run-vstrt",
                            *common,
                            f"VSTRT_OUTPUT_DIR={output_root}/vstrt",
                        ),
                    ),
                    Step(
                        key=f"smoke:{selection.key}:vsgan",
                        label=f"Smoke-test VSGAN on {selection.key}",
                        command=_make(
                            benchmark_dir,
                            "run-vsgan",
                            *common,
                            f"VSGAN_OUTPUT_DIR={output_root}/vsgan",
                        ),
                    ),
                )
            )
    return steps


def _project_steps(
    *,
    root: Path,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    return [
        Step(
            key=f"project:{selection.key}",
            label=f"Run project benchmark on {selection.key}",
            command=_make(
                benchmark_dir,
                "run-project",
                *_selection_variables(selection, options.gpu_id),
            ),
        )
        for selection in selections
    ]


def _comparative_steps(
    *,
    root: Path,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = []
    for selection in selections:
        variables = (
            *_selection_variables(selection, options.gpu_id),
            f"VAPOURSYNTH_MODE={options.mode}",
        )
        steps.append(
            Step(
                key=f"quality:{options.mode}:{selection.key}",
                label=f"Run {options.mode} quality gates on {selection.key}",
                command=_make(
                    benchmark_dir,
                    "quality-gates",
                    *variables,
                ),
            )
        )
    for selection in selections:
        variables = (
            *_selection_variables(selection, options.gpu_id),
            f"VAPOURSYNTH_MODE={options.mode}",
            f"RESUME={int(options.resume)}",
        )
        steps.append(
            Step(
                key=f"campaign:{options.mode}:{selection.key}",
                label=f"Run {options.mode} comparative campaign on {selection.key}",
                command=_make(
                    benchmark_dir,
                    "run-comparative",
                    *variables,
                ),
            )
        )
    return steps


def _tuned_steps(
    *,
    root: Path,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = []
    for target, stage in (
        ("run-tuned-sweep", "sweep"),
        ("run-tuned-quality", "quality"),
        ("run-tuned-campaign", "campaign"),
    ):
        for selection in selections:
            variables = list(_selection_variables(selection, options.gpu_id))
            variables.append(
                f"TUNING_CONTRACT={selection.workload.tuning_contract}"
            )
            if stage == "sweep":
                variables.append(f"TUNING_RESUME={int(options.resume)}")
            elif stage == "campaign":
                variables.append(f"RESUME={int(options.resume)}")
            steps.append(
                Step(
                    key=f"tuned:{stage}:{selection.key}",
                    label=f"Run tuned {stage} on {selection.key}",
                    command=_make(benchmark_dir, target, *variables),
                )
            )

    selected_variants: dict[str, set[str]] = {}
    for selection in selections:
        selected_variants.setdefault(selection.workload.key, set()).add(
            selection.variant.name
        )
    for workload in (selection.workload for selection in selections):
        variants = selected_variants[workload.key]
        if variants != {"720p", "1080p"}:
            continue
        key = f"tuned:matrix:{workload.key}"
        if any(step.key == key for step in steps):
            continue
        steps.append(
            Step(
                key=key,
                label=f"Verify tuned publication matrix for {workload.key}",
                command=_make(
                    benchmark_dir,
                    "verify-tuned-matrix",
                    f"MANIFEST={workload.manifest}",
                ),
            )
        )
    return steps


def _diagnostic_steps(
    *,
    root: Path,
    matrix: WorkflowMatrix,
    options: WorkflowOptions,
    selections: Sequence[Selection],
) -> list[Step]:
    benchmark_dir = root / "benchmarks"
    steps = [
        Step(
            key=f"diagnostics:trtexec:{selection.key}",
            label=f"Run trtexec ceiling on {selection.key}",
            command=_make(
                benchmark_dir,
                "run-trtexec",
                *_selection_variables(selection, options.gpu_id),
            ),
        )
        for selection in selections
    ]
    for selection in selections:
        if (
            selection.workload.key == matrix.nsight_workload
            and selection.variant.name == matrix.nsight_variant
        ):
            steps.append(
                Step(
                    key=f"diagnostics:nsight:{selection.key}",
                    label=f"Capture Nsight trace on {selection.key}",
                    command=_make(
                        benchmark_dir,
                        "profile-nsight",
                        *_selection_variables(selection, options.gpu_id),
                    ),
                )
            )
    return steps


def build_plan(
    *,
    root: Path,
    matrix: WorkflowMatrix,
    options: WorkflowOptions,
) -> tuple[Step, ...]:
    """Build the complete ordered plan for one goal and matrix selection."""
    options.validate()
    selections = matrix.select(
        workload_key=options.workload_key,
        variant_name=options.variant_name,
    )
    steps = [
        *_build_steps(root=root, options=options),
        *_asset_steps(root=root, selections=selections),
        *_engine_steps(root=root, options=options, selections=selections),
        *_smoke_steps(root=root, options=options, selections=selections),
    ]
    if options.goal == "project":
        steps.extend(
            _project_steps(
                root=root,
                options=options,
                selections=selections,
            )
        )
    elif options.goal == "comparative":
        steps.extend(
            _comparative_steps(
                root=root,
                options=options,
                selections=selections,
            )
        )
    elif options.goal == "tuned":
        steps.extend(
            _tuned_steps(
                root=root,
                options=options,
                selections=selections,
            )
        )
    else:
        steps.extend(
            _diagnostic_steps(
                root=root,
                matrix=matrix,
                options=options,
                selections=selections,
            )
        )
    keys = [step.key for step in steps]
    if len(keys) != len(set(keys)):
        raise WorkflowError("Workflow plan contains duplicate step keys")
    return tuple(steps)


@dataclass
class WorkflowState:
    """Append-only logical completion state for one exact workflow context."""

    path: Path
    context: dict[str, Any]
    completed_steps: list[dict[str, str]]

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        context: dict[str, Any],
        resume: bool,
    ) -> WorkflowState:
        if not path.exists():
            if resume:
                raise WorkflowError(f"Resume state does not exist: {path}")
            return cls(path=path, context=context, completed_steps=[])
        if not resume:
            raise WorkflowError(
                f"Workflow state exists; use --resume or remove it: {path}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"Cannot read workflow state {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise WorkflowError(f"Invalid workflow state: {path}")
        if value.get("context") != context:
            raise WorkflowError("Workflow state does not match current selection")
        completed = value.get("completed_steps")
        if not isinstance(completed, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("key"), str)
            and isinstance(item.get("completed_at"), str)
            for item in completed
        ):
            raise WorkflowError("Workflow state has invalid completed steps")
        return cls(path=path, context=context, completed_steps=completed)

    @property
    def completed_keys(self) -> set[str]:
        return {item["key"] for item in self.completed_steps}

    def complete(self, key: str) -> None:
        if key in self.completed_keys:
            return
        self.completed_steps.append(
            {
                "key": key,
                # datetime.UTC is unavailable on the supported Python 3.10 host.
                "completed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            }
        )
        document = {
            "schema_version": 1,
            "context": self.context,
            "completed_steps": self.completed_steps,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


CommandExecutor = Callable[[Sequence[str], Path], None]


def execute_command(command: Sequence[str], cwd: Path) -> None:
    """Execute one low-level command without a shell."""
    subprocess.run(command, cwd=cwd, check=True)


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_plan(
    plan: Sequence[Step],
    *,
    root: Path,
    state: WorkflowState | None,
    dry_run: bool,
    executor: CommandExecutor = execute_command,
) -> None:
    """Execute or print a workflow plan, persisting only successful steps."""
    completed = state.completed_keys if state is not None else set()
    for index, step in enumerate(plan, start=1):
        prefix = f"[{index}/{len(plan)}]"
        if step.key in completed:
            print(f"{prefix} SKIP {step.label}", flush=True)
            continue
        print(f"{prefix} {step.label}", flush=True)
        print(f"  {_format_command(step.command)}", flush=True)
        if dry_run:
            continue
        executor(step.command, root)
        assert state is not None
        state.complete(step.key)
