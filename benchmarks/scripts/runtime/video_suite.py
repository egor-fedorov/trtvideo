"""Shared execution core for full-video benchmark suites."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.scripts.runtime.cpu import snapshot_child_cpu, summarize_child_cpu
from benchmarks.scripts.runtime.environment import (
    environment_errors,
    relative_artifact_path,
    sha256_file,
    write_json,
)
from benchmarks.scripts.runtime.nvml import NvmlSampler, summarize_samples, write_samples
from benchmarks.scripts.runtime.suite import (
    RunExecutor,
    SuitePolicy,
    SuiteRunner,
    canonical_suite_errors,
    report_publishability_errors,
    suite_publishability_errors,
)
from trtvideo.benchmarking.lifecycle import (
    FrameLifecycleMarkers,
    LifecycleTimingError,
    median_detailed_phase_intervals,
    summarize_lifecycle,
)
from trtvideo.benchmarking.validation import OutputContract, validate_output

ManifestCommand = list[str] | list[list[str]]
ProcessExecutor = Callable[[Path, Path], "ProcessResult"]
LifecycleReader = Callable[["ProcessResult"], FrameLifecycleMarkers]
OutputValidator = Callable[[Path, OutputContract], dict[str, Any]]
SuiteExecutorFactory = Callable[[NvmlSampler, dict[str, Any]], RunExecutor]


@dataclass(frozen=True)
class ProcessResult:
    """Exit status and monotonic boundaries of one measured process tree."""

    returncode: int
    process_started_ns: int
    process_finished_ns: int


@dataclass(frozen=True)
class VideoRunPaths:
    """Canonical artifact layout shared by every full-video implementation."""

    run_dir: Path
    warmup_output: Path
    measured_output: Path
    warmup_stdout: Path
    warmup_stderr: Path
    measured_stdout: Path
    measured_stderr: Path
    nvml_samples: Path
    manifest: Path

    @classmethod
    def create(cls, output_dir: Path, run_index: int) -> VideoRunPaths:
        run_dir = output_dir / f"run-{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            warmup_output=run_dir / "warmup.mp4",
            measured_output=run_dir / "output.mp4",
            warmup_stdout=run_dir / "warmup.stdout.log",
            warmup_stderr=run_dir / "warmup.stderr.log",
            measured_stdout=run_dir / "measured.stdout.log",
            measured_stderr=run_dir / "measured.stderr.log",
            nvml_samples=run_dir / "nvml.samples.jsonl",
            manifest=run_dir / "manifest.json",
        )

    def manifest_artifacts(
        self,
        root: Path,
        extra: Mapping[str, Path] | None = None,
    ) -> dict[str, str]:
        artifacts = {
            "manifest": relative_artifact_path(self.manifest, root),
            "warmup_stdout": relative_artifact_path(self.warmup_stdout, root),
            "warmup_stderr": relative_artifact_path(self.warmup_stderr, root),
            "measured_stdout": relative_artifact_path(self.measured_stdout, root),
            "measured_stderr": relative_artifact_path(self.measured_stderr, root),
            "nvml_samples": relative_artifact_path(self.nvml_samples, root),
        }
        artifacts.update(
            {name: relative_artifact_path(path, root) for name, path in (extra or {}).items()}
        )
        return artifacts


@dataclass(frozen=True)
class ProcessInvocation:
    """Manifest command and executable adapter for one process invocation."""

    command: ManifestCommand
    execute: ProcessExecutor


@dataclass(frozen=True)
class VideoRunSpec:
    """Implementation-independent contract for one warmup and measured run."""

    run_index: int
    frames: int
    warmup_frames: int
    keep_outputs: bool
    max_compute_processes: int
    max_graphics_processes: int
    require_reproducible_environment: bool
    manifest_fields: dict[str, Any]
    warmup: ProcessInvocation
    measured: ProcessInvocation
    warmup_contract: OutputContract
    measured_contract: OutputContract
    lifecycle_reader: LifecycleReader
    extra_artifacts: Mapping[str, Path] | None = None


@dataclass(frozen=True)
class VideoSuiteSpec:
    """Shared repeat, publication, and summary contract for one implementation."""

    output_dir: Path
    policy: SuitePolicy
    label: str
    frames: int
    warmup_frames: int
    sample_interval_ms: int
    gpu_id: int
    benchmark_contract: Mapping[str, Any] | None
    parameter_fields: dict[str, Any]
    summary_fields: dict[str, Any]
    include_median_lifecycle: bool = False


def asset_record(kind: str, path: Path, root: Path) -> dict[str, Any]:
    """Hash one required benchmark asset using a privacy-safe path."""
    if not path.is_file():
        raise FileNotFoundError(f"Required {kind} asset not found: {path}")
    return {
        "kind": kind,
        "path": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_existing_output(
    path: Path,
    contract: OutputContract,
    *,
    missing_message: str,
    validate: OutputValidator,
) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "errors": [missing_message]}
    return validate(path, contract)


def _cleanup_output(path: Path, *, keep: bool) -> None:
    if not keep and path.exists():
        path.unlink()


def run_video_measurement(
    spec: VideoRunSpec,
    *,
    paths: VideoRunPaths,
    sampler: NvmlSampler,
    root: Path,
    validate: OutputValidator = validate_output,
) -> dict[str, Any]:
    """Execute and validate one discarded warmup plus one measured process."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_index": spec.run_index,
        **spec.manifest_fields,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "commands": {
            "warmup": spec.warmup.command,
            "measured": spec.measured.command,
        },
        "artifacts": paths.manifest_artifacts(root, spec.extra_artifacts),
        "errors": [],
    }

    warmup_result = spec.warmup.execute(paths.warmup_stdout, paths.warmup_stderr)
    warmup_validation = _validate_existing_output(
        paths.warmup_output,
        spec.warmup_contract,
        missing_message="Warmup output was not created",
        validate=validate,
    )
    manifest["warmup"] = {
        "returncode": warmup_result.returncode,
        "validation": warmup_validation,
    }
    if warmup_result.returncode != 0:
        manifest["errors"].append(f"Warmup process exited with code {warmup_result.returncode}")
    if not warmup_validation.get("valid"):
        manifest["errors"].extend(warmup_validation.get("errors", []))
    if manifest["errors"]:
        manifest["status"] = "invalid"
        write_json(paths.manifest, manifest)
        return manifest
    _cleanup_output(paths.warmup_output, keep=spec.keep_outputs)

    cpu_before = snapshot_child_cpu()
    sampler.start(time.perf_counter())
    try:
        measured_result = spec.measured.execute(
            paths.measured_stdout,
            paths.measured_stderr,
        )
    finally:
        cpu_after = snapshot_child_cpu()
        samples = sampler.stop()
    wall_time_sec = (
        measured_result.process_finished_ns - measured_result.process_started_ns
    ) / 1_000_000_000
    relative_samples = sampler.samples_relative_to(
        samples,
        measured_result.process_started_ns / 1_000_000_000,
    )
    cpu_summary = summarize_child_cpu(
        cpu_before,
        cpu_after,
        wall_time_sec=wall_time_sec,
    ).as_dict()
    write_samples(paths.nvml_samples, relative_samples)
    nvml_summary = summarize_samples(
        relative_samples,
        wall_time_sec=wall_time_sec,
        frames=spec.frames,
        max_compute_processes=spec.max_compute_processes,
        max_graphics_processes=spec.max_graphics_processes,
    )
    measured_validation = _validate_existing_output(
        paths.measured_output,
        spec.measured_contract,
        missing_message="Measured output was not created",
        validate=validate,
    )
    output = (
        asset_record("output", paths.measured_output, root)
        if paths.measured_output.is_file()
        else None
    )
    lifecycle_summary = None
    lifecycle_error = None
    try:
        lifecycle_summary = summarize_lifecycle(
            process_started_ns=measured_result.process_started_ns,
            process_finished_ns=measured_result.process_finished_ns,
            markers=spec.lifecycle_reader(measured_result),
            expected_frames=spec.frames,
        )
    except LifecycleTimingError as exc:
        lifecycle_error = str(exc)

    manifest["measured"] = {
        "returncode": measured_result.returncode,
        "metrics": {
            "wall_time_sec": wall_time_sec,
            "end_to_end_fps": spec.frames / wall_time_sec if wall_time_sec > 0 else None,
            "processed_frames": spec.frames,
            "cpu": cpu_summary,
            "lifecycle": lifecycle_summary,
            "nvml": nvml_summary,
        },
        "validation": measured_validation,
        "output": output,
    }
    if measured_result.returncode != 0:
        manifest["errors"].append(f"Measured process exited with code {measured_result.returncode}")
    if lifecycle_error is not None:
        manifest["errors"].append(f"Lifecycle timing: {lifecycle_error}")
    if not measured_validation.get("valid"):
        manifest["errors"].extend(measured_validation.get("errors", []))
    if not nvml_summary.get("valid"):
        manifest["errors"].extend(nvml_summary.get("errors", []))

    reproducibility_errors = environment_errors(spec.manifest_fields["environment"])
    manifest["reproducibility"] = {
        "publishable": not reproducibility_errors,
        "errors": reproducibility_errors,
    }
    if spec.require_reproducible_environment:
        manifest["errors"].extend(reproducibility_errors)
    manifest["status"] = "valid" if not manifest["errors"] else "invalid"
    write_json(paths.manifest, manifest)
    if manifest["status"] == "valid":
        _cleanup_output(paths.measured_output, keep=spec.keep_outputs)
    return manifest


def _end_to_end_fps(manifest: dict[str, Any]) -> float:
    value = manifest.get("measured", {}).get("metrics", {}).get("end_to_end_fps")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Valid video run has no end_to_end_fps metric")
    return float(value)


def _power_limit(manifest: dict[str, Any]) -> float | None:
    value = (
        manifest.get("measured", {})
        .get("metrics", {})
        .get("nvml", {})
        .get("power", {})
        .get("limit_w")
    )
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Video run has an invalid GPU power limit")
    return float(value)


def run_video_suite(
    spec: VideoSuiteSpec,
    executor_factory: SuiteExecutorFactory,
) -> tuple[dict[str, Any], int]:
    """Execute a sampled repeated suite and write its acceptance summary."""
    sampler = NvmlSampler(spec.gpu_id, spec.sample_interval_ms)
    gpu = sampler.initialize()
    try:
        execute_run = executor_factory(sampler, gpu)
        suite_result = SuiteRunner(
            spec.policy,
            label=spec.label,
            frames=spec.frames,
            metric_reader=_end_to_end_fps,
            power_limit_reader=_power_limit,
        ).execute(execute_run)
    finally:
        sampler.shutdown()

    run_manifests = list(suite_result.runs)
    statistics = dict(suite_result.statistics)
    if spec.include_median_lifecycle:
        lifecycle_summaries = [
            run.get("measured", {}).get("metrics", {}).get("lifecycle") for run in run_manifests
        ]
        statistics["median_lifecycle_intervals_sec"] = (
            median_detailed_phase_intervals(lifecycle_summaries)
            if lifecycle_summaries
            and all(isinstance(summary, dict) for summary in lifecycle_summaries)
            else {}
        )

    parameters = {
        "frames": spec.frames,
        "warmup_frames": spec.warmup_frames,
        "initial_runs": spec.policy.initial_runs,
        "extra_runs_on_spread": spec.policy.extra_runs,
        "spread_threshold": spec.policy.spread_threshold,
        "max_relative_spread": (
            spec.policy.max_relative_spread
            if spec.policy.max_relative_spread is not None
            else spec.policy.spread_threshold
        ),
        "idle_seconds": spec.policy.idle_seconds,
        "nvml_sample_interval_ms": spec.sample_interval_ms,
        **spec.parameter_fields,
    }
    canonical_errors = canonical_suite_errors(
        parameters,
        spec.benchmark_contract,
        include_warmup_frames=True,
    )
    publishability_errors = suite_publishability_errors(
        status=suite_result.status,
        canonical_errors=canonical_errors,
        runs=run_manifests,
        acceptance_only=True,
    )
    summary = {
        "schema_version": 1,
        "status": suite_result.status,
        "scope": "acceptance",
        "publishable": not publishability_errors,
        "publishability": {
            "canonical_contract": not canonical_errors,
            "errors": publishability_errors,
        },
        **spec.summary_fields,
        "parameters": parameters,
        "statistics": statistics,
        "errors": list(suite_result.errors),
        "runs": [
            {
                "index": run["run_index"],
                "status": run["status"],
                "manifest": run["artifacts"]["manifest"],
                "end_to_end_fps": run.get("measured", {}).get("metrics", {}).get("end_to_end_fps"),
            }
            for run in run_manifests
        ],
    }
    write_json(spec.output_dir / "suite.json", summary)
    print(
        f"Benchmark suite {suite_result.status}: "
        f"median={statistics['median_fps']!r} FPS, "
        f"spread={statistics['relative_spread']!r}",
        file=sys.stderr,
    )
    report_publishability_errors(
        publishability_errors,
        acceptance_only=True,
    )
    return summary, 0 if suite_result.status == "valid" else 2
