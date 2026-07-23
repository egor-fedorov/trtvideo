"""External process benchmark suite for full-video implementations."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_media.benchmarking.cpu import snapshot_child_cpu, summarize_child_cpu
from ai_media.benchmarking.environment import (
    collect_environment,
    environment_errors,
    relative_artifact_path,
    sanitize_command,
    sha256_file,
    write_json,
)
from ai_media.benchmarking.nvml import NvmlSampler, summarize_samples, write_samples
from ai_media.benchmarking.suite import (
    SuitePolicy,
    SuiteRunner,
    canonical_suite_errors,
    suite_publishability_errors,
)
from ai_media.benchmarking.validation import OutputContract, validate_output
from benchmarks.scripts.competitor_common import CommandSpec, CompetitorError

CommandFactory = Callable[[Path, int], CommandSpec]


@dataclass(frozen=True)
class ExternalImplementation:
    """Identity and process constraints of one external product."""

    product: str
    backend: str
    comparison_class: str
    metadata: dict[str, Any]
    max_compute_processes: int
    max_graphics_processes: int


@dataclass(frozen=True)
class ExternalVideoWorkload:
    """Canonical assets and output contract for one video workload."""

    workload_id: str
    variant: str
    output_dir: Path
    frames: int
    warmup_frames: int
    output_contract: dict[str, Any]
    benchmark_contract: dict[str, Any]
    assets: dict[str, Path]


@dataclass(frozen=True)
class ExternalVideoSuiteConfig:
    """Composition root for one external full-video benchmark."""

    implementation: ExternalImplementation
    workload: ExternalVideoWorkload
    policy: SuitePolicy
    sample_interval_ms: int
    gpu_id: int
    implementation_parameters: dict[str, Any]
    warmup_command: CommandFactory
    measured_command: CommandFactory
    keep_outputs: bool = False


def _asset_record(kind: str, path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompetitorError(f"Required {kind} asset not found: {path}")
    return {
        "kind": kind,
        "path": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sanitize_spec(spec: CommandSpec, root: Path) -> list[list[str]]:
    return [sanitize_command(command, root) for command in spec]


def run_command_spec(
    spec: CommandSpec,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    """Run an argv-only command pipeline without invoking a shell."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    previous_stdout = None
    try:
        with (
            stdout_path.open("wb") as stdout,
            stderr_path.open("wb") as stderr,
        ):
            for index, command in enumerate(spec):
                is_last = index == len(spec) - 1
                process = subprocess.Popen(
                    command,
                    stdin=previous_stdout,
                    stdout=stdout if is_last else subprocess.PIPE,
                    stderr=stderr,
                )
                if previous_stdout is not None:
                    previous_stdout.close()
                previous_stdout = process.stdout
                processes.append(process)

            returncodes = [process.wait() for process in reversed(processes)]
            returncodes.reverse()
    except OSError as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        stderr_path.write_text(f"Failed to start pipeline: {exc}\n", encoding="utf-8")
        return 127
    return next((code for code in returncodes if code != 0), 0)


def _environment(config: ExternalVideoSuiteConfig, gpu: dict[str, Any]) -> dict[str, Any]:
    environment = collect_environment(gpu)
    environment["image"] = {
        "reference": os.environ.get(
            "AI_MEDIA_IMAGE_REF", config.implementation.metadata["image"]
        ),
        "id": os.environ.get("AI_MEDIA_IMAGE_ID", "unknown"),
        "base_reference": os.environ.get("AI_MEDIA_BASE_IMAGE", "unknown"),
        "repository_revision": os.environ.get("AI_MEDIA_BUILD_REVISION", "unknown"),
        "source_dirty": os.environ.get("AI_MEDIA_BUILD_DIRTY", "unknown"),
    }
    environment["implementation"] = config.implementation.metadata
    return environment


def _contract(config: ExternalVideoSuiteConfig, frames: int, *, bitrate: bool) -> OutputContract:
    values = dict(config.workload.output_contract)
    values["frames"] = frames
    if not bitrate:
        values["target_bitrate_mbps"] = None
    return OutputContract(**values)


def _cleanup(path: Path, keep: bool) -> None:
    if not keep and path.exists():
        path.unlink()


def _run_one(
    config: ExternalVideoSuiteConfig,
    *,
    run_index: int,
    sampler: NvmlSampler,
    environment: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    workload = config.workload
    implementation = config.implementation
    run_dir = workload.output_dir / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = run_dir / "warmup.mp4"
    measured_output = run_dir / "output.mp4"
    warmup_stdout = run_dir / "warmup.stdout.log"
    warmup_stderr = run_dir / "warmup.stderr.log"
    measured_stdout = run_dir / "measured.stdout.log"
    measured_stderr = run_dir / "measured.stderr.log"
    samples_path = run_dir / "nvml.samples.jsonl"
    manifest_path = run_dir / "manifest.json"
    warmup_spec = config.warmup_command(warmup_output, workload.warmup_frames)
    measured_spec = config.measured_command(measured_output, workload.frames)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_index": run_index,
        "product": implementation.product,
        "backend": implementation.backend,
        "comparison_class": implementation.comparison_class,
        "workload_id": workload.workload_id,
        "variant": workload.variant,
        "implementation": implementation.metadata,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "frames": workload.frames,
            "warmup_frames": workload.warmup_frames,
            "gpu_id": config.gpu_id,
            "nvml_sample_interval_ms": config.sample_interval_ms,
            "max_compute_processes": implementation.max_compute_processes,
            "max_graphics_processes": implementation.max_graphics_processes,
            **config.implementation_parameters,
        },
        "commands": {
            "warmup": _sanitize_spec(warmup_spec, root),
            "measured": _sanitize_spec(measured_spec, root),
        },
        "assets": assets,
        "environment": environment,
        "artifacts": {
            "manifest": relative_artifact_path(manifest_path, root),
            "warmup_stdout": relative_artifact_path(warmup_stdout, root),
            "warmup_stderr": relative_artifact_path(warmup_stderr, root),
            "measured_stdout": relative_artifact_path(measured_stdout, root),
            "measured_stderr": relative_artifact_path(measured_stderr, root),
            "nvml_samples": relative_artifact_path(samples_path, root),
        },
        "errors": [],
    }

    warmup_returncode = run_command_spec(warmup_spec, warmup_stdout, warmup_stderr)
    warmup_validation = (
        validate_output(
            warmup_output,
            _contract(config, workload.warmup_frames, bitrate=False),
        )
        if warmup_output.is_file()
        else {"valid": False, "errors": ["Warmup output was not created"]}
    )
    manifest["warmup"] = {
        "returncode": warmup_returncode,
        "validation": warmup_validation,
    }
    if warmup_returncode != 0:
        manifest["errors"].append(f"Warmup process exited with code {warmup_returncode}")
    if not warmup_validation.get("valid"):
        manifest["errors"].extend(warmup_validation.get("errors", []))
    if manifest["errors"]:
        manifest["status"] = "invalid"
        write_json(manifest_path, manifest)
        return manifest
    _cleanup(warmup_output, config.keep_outputs)

    cpu_before = snapshot_child_cpu()
    sampler.start(time.perf_counter())
    start_time = time.perf_counter()
    try:
        measured_returncode = run_command_spec(
            measured_spec,
            measured_stdout,
            measured_stderr,
        )
    finally:
        end_time = time.perf_counter()
        cpu_after = snapshot_child_cpu()
        samples = sampler.samples_relative_to(sampler.stop(), start_time)
    wall_time_sec = end_time - start_time
    cpu_summary = summarize_child_cpu(
        cpu_before,
        cpu_after,
        wall_time_sec=wall_time_sec,
    ).as_dict()
    write_samples(samples_path, samples)
    nvml_summary = summarize_samples(
        samples,
        wall_time_sec=wall_time_sec,
        frames=workload.frames,
        max_compute_processes=implementation.max_compute_processes,
        max_graphics_processes=implementation.max_graphics_processes,
    )
    measured_validation = (
        validate_output(measured_output, _contract(config, workload.frames, bitrate=True))
        if measured_output.is_file()
        else {"valid": False, "errors": ["Measured output was not created"]}
    )
    output_asset = (
        _asset_record("output", measured_output, root) if measured_output.is_file() else None
    )
    manifest["measured"] = {
        "returncode": measured_returncode,
        "metrics": {
            "wall_time_sec": wall_time_sec,
            "end_to_end_fps": workload.frames / wall_time_sec if wall_time_sec > 0 else None,
            "processed_frames": workload.frames,
            "cpu": cpu_summary,
            "nvml": nvml_summary,
        },
        "validation": measured_validation,
        "output": output_asset,
    }
    if measured_returncode != 0:
        manifest["errors"].append(
            f"Measured process exited with code {measured_returncode}"
        )
    if not measured_validation.get("valid"):
        manifest["errors"].extend(measured_validation.get("errors", []))
    if not nvml_summary.get("valid"):
        manifest["errors"].extend(nvml_summary.get("errors", []))
    reproducibility_errors = environment_errors(environment)
    manifest["reproducibility"] = {
        "publishable": not reproducibility_errors,
        "errors": reproducibility_errors,
    }
    manifest["errors"].extend(reproducibility_errors)
    manifest["status"] = "valid" if not manifest["errors"] else "invalid"
    write_json(manifest_path, manifest)
    if manifest["status"] == "valid":
        _cleanup(measured_output, config.keep_outputs)
    return manifest


def _end_to_end_fps(manifest: dict[str, Any]) -> float:
    value = manifest.get("measured", {}).get("metrics", {}).get("end_to_end_fps")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompetitorError("Valid external run has no end_to_end_fps metric")
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
        raise CompetitorError("External run has an invalid GPU power limit")
    return float(value)


def run_external_video_suite(
    config: ExternalVideoSuiteConfig,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run an external full pipeline with the common 3+2 contract."""
    root = (root or Path.cwd()).resolve()
    workload = config.workload
    implementation = config.implementation
    workload.output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        name: _asset_record(name, path, root) for name, path in workload.assets.items()
    }
    sampler = NvmlSampler(config.gpu_id, config.sample_interval_ms)
    gpu = sampler.initialize()
    environment = _environment(config, gpu)
    summary_path = workload.output_dir / "suite.json"
    suite_runner = SuiteRunner(
        config.policy,
        label=implementation.product,
        frames=workload.frames,
        metric_reader=_end_to_end_fps,
        power_limit_reader=_power_limit,
    )

    def execute_run(run_index: int) -> dict[str, Any]:
        return _run_one(
            config,
            run_index=run_index,
            sampler=sampler,
            environment=environment,
            assets=assets,
            root=root,
        )

    try:
        suite_result = suite_runner.execute(execute_run)
    finally:
        sampler.shutdown()

    run_manifests = list(suite_result.runs)
    statistics = suite_result.statistics
    spread = statistics["relative_spread"]
    status = suite_result.status
    parameters = {
        "frames": workload.frames,
        "warmup_frames": workload.warmup_frames,
        "initial_runs": config.policy.initial_runs,
        "extra_runs_on_spread": config.policy.extra_runs,
        "spread_threshold": config.policy.spread_threshold,
        "idle_seconds": config.policy.idle_seconds,
        "nvml_sample_interval_ms": config.sample_interval_ms,
        "max_compute_processes": implementation.max_compute_processes,
        "max_graphics_processes": implementation.max_graphics_processes,
        **config.implementation_parameters,
    }
    canonical_errors = canonical_suite_errors(
        parameters,
        workload.benchmark_contract,
        include_warmup_frames=True,
    )
    publishability_errors = suite_publishability_errors(
        status=status,
        canonical_errors=canonical_errors,
        runs=run_manifests,
        acceptance_only=True,
    )
    summary = {
        "schema_version": 1,
        "document_type": "benchmark-result",
        "status": status,
        "scope": "acceptance",
        "publishable": not publishability_errors,
        "publishability": {
            "canonical_contract": not canonical_errors,
            "errors": publishability_errors,
        },
        "product": implementation.product,
        "backend": implementation.backend,
        "comparison_class": implementation.comparison_class,
        "workload_id": workload.workload_id,
        "variant": workload.variant,
        "implementation": implementation.metadata,
        "parameters": parameters,
        "statistics": statistics,
        "errors": list(suite_result.errors),
        "runs": [
            {
                "index": run["run_index"],
                "status": run["status"],
                "manifest": run["artifacts"]["manifest"],
                "end_to_end_fps": run.get("measured", {})
                .get("metrics", {})
                .get("end_to_end_fps"),
            }
            for run in run_manifests
        ],
    }
    write_json(summary_path, summary)
    print(
        f"Benchmark suite {status}: median={statistics['median_fps']!r} FPS, "
        f"spread={spread!r}",
        file=sys.stderr,
    )
    if publishability_errors:
        print("Benchmark suite is not publishable:", file=sys.stderr)
        for error in publishability_errors:
            print(f"  - {error}", file=sys.stderr)
    return summary, 0 if status == "valid" else 2
