"""External process benchmark suite for full-video competitor pipelines."""

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

from ai_media.benchmarking.environment import (
    collect_environment,
    environment_errors,
    relative_artifact_path,
    sanitize_command,
    sha256_file,
    write_json,
)
from ai_media.benchmarking.nvml import NvmlSampler, summarize_samples, write_samples
from ai_media.benchmarking.runner import compute_suite_statistics, should_extend_suite
from ai_media.benchmarking.validation import OutputContract, validate_output
from benchmarks.scripts.competitor_common import CommandSpec, CompetitorError

CommandFactory = Callable[[Path, int], CommandSpec]


@dataclass(frozen=True)
class ExternalVideoSuiteConfig:
    """Configuration for one competitor's full video path."""

    product: str
    backend: str
    comparison_class: str
    implementation: dict[str, Any]
    workload_id: str
    variant: str
    input_path: Path
    output_dir: Path
    frames: int
    warmup_frames: int
    initial_runs: int
    extra_runs: int
    spread_threshold: float
    idle_seconds: float
    sample_interval_ms: int
    gpu_id: int
    output_contract: dict[str, Any]
    assets: dict[str, Path]
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
        "reference": os.environ.get("AI_MEDIA_IMAGE_REF", config.implementation["image"]),
        "id": os.environ.get("AI_MEDIA_IMAGE_ID", "unknown"),
        "base_reference": os.environ.get("AI_MEDIA_BASE_IMAGE", "unknown"),
        "repository_revision": os.environ.get("AI_MEDIA_BUILD_REVISION", "unknown"),
        "source_dirty": os.environ.get("AI_MEDIA_BUILD_DIRTY", "unknown"),
    }
    environment["competitor"] = config.implementation
    return environment


def _contract(config: ExternalVideoSuiteConfig, frames: int, *, bitrate: bool) -> OutputContract:
    values = dict(config.output_contract)
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
    run_dir = config.output_dir / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = run_dir / "warmup.mp4"
    measured_output = run_dir / "output.mp4"
    warmup_stdout = run_dir / "warmup.stdout.log"
    warmup_stderr = run_dir / "warmup.stderr.log"
    measured_stdout = run_dir / "measured.stdout.log"
    measured_stderr = run_dir / "measured.stderr.log"
    samples_path = run_dir / "nvml.samples.jsonl"
    manifest_path = run_dir / "manifest.json"
    warmup_spec = config.warmup_command(warmup_output, config.warmup_frames)
    measured_spec = config.measured_command(measured_output, config.frames)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_index": run_index,
        "product": config.product,
        "backend": config.backend,
        "comparison_class": config.comparison_class,
        "workload_id": config.workload_id,
        "variant": config.variant,
        "implementation": config.implementation,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "frames": config.frames,
            "warmup_frames": config.warmup_frames,
            "gpu_id": config.gpu_id,
            "nvml_sample_interval_ms": config.sample_interval_ms,
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
            _contract(config, config.warmup_frames, bitrate=False),
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
        samples = sampler.samples_relative_to(sampler.stop(), start_time)
    wall_time_sec = end_time - start_time
    write_samples(samples_path, samples)
    nvml_summary = summarize_samples(
        samples,
        wall_time_sec=wall_time_sec,
        frames=config.frames,
    )
    measured_validation = (
        validate_output(measured_output, _contract(config, config.frames, bitrate=True))
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
            "end_to_end_fps": config.frames / wall_time_sec if wall_time_sec > 0 else None,
            "processed_frames": config.frames,
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


def run_external_video_suite(
    config: ExternalVideoSuiteConfig,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run a competitor's full pipeline with the common 3+2 contract."""
    root = (root or Path.cwd()).resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        name: _asset_record(name, path, root) for name, path in config.assets.items()
    }
    sampler = NvmlSampler(config.gpu_id, config.sample_interval_ms)
    gpu = sampler.initialize()
    environment = _environment(config, gpu)
    run_manifests: list[dict[str, Any]] = []
    target_runs = config.initial_runs
    summary_path = config.output_dir / "suite.json"

    try:
        run_index = 1
        while run_index <= target_runs:
            if run_index > 1 and config.idle_seconds > 0:
                time.sleep(config.idle_seconds)
            print(
                f"Benchmark run {run_index}/{target_runs}: {config.product}, "
                f"{config.frames} frames",
                file=sys.stderr,
            )
            result = _run_one(
                config,
                run_index=run_index,
                sampler=sampler,
                environment=environment,
                assets=assets,
                root=root,
            )
            run_manifests.append(result)
            if result.get("status") != "valid":
                break
            if run_index == config.initial_runs and config.extra_runs > 0:
                fps_values = [
                    run["measured"]["metrics"]["end_to_end_fps"]
                    for run in run_manifests
                ]
                if should_extend_suite(fps_values, config.spread_threshold):
                    target_runs += config.extra_runs
            run_index += 1
    finally:
        sampler.shutdown()

    valid_runs = [run for run in run_manifests if run.get("status") == "valid"]
    fps_values = [
        run["measured"]["metrics"]["end_to_end_fps"] for run in valid_runs
    ]
    statistics = compute_suite_statistics(fps_values)
    spread = statistics["relative_spread"]
    all_valid = len(valid_runs) == len(run_manifests) == target_runs
    stable = all_valid and spread is not None and spread <= config.spread_threshold
    status = "valid" if stable else ("unstable" if all_valid else "invalid")
    summary = {
        "schema_version": 1,
        "document_type": "benchmark-result",
        "status": status,
        "publishable": stable
        and all(run["reproducibility"]["publishable"] for run in valid_runs),
        "product": config.product,
        "backend": config.backend,
        "comparison_class": config.comparison_class,
        "workload_id": config.workload_id,
        "variant": config.variant,
        "implementation": config.implementation,
        "statistics": statistics,
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
    return summary, 0 if status == "valid" else 2
