"""TheAnimeScripter adapter for the shared full-video measurement core."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.benchmark import CompetitorError, load_json
from benchmarks.scripts.runners.tas_profile import TasExecutionProfile
from benchmarks.scripts.runners.tas_runtime import adapter_sha256
from benchmarks.scripts.runtime.command import CommandSpec
from benchmarks.scripts.runtime.environment import (
    collect_environment,
    collect_image_identity,
    sanitize_command,
    write_json,
)
from benchmarks.scripts.runtime.suite import SuitePolicy
from benchmarks.scripts.runtime.video_suite import (
    ProcessInvocation,
    ProcessResult,
    VideoRunPaths,
    VideoRunSpec,
    VideoSuiteSpec,
    asset_record,
    run_video_measurement,
    run_video_suite,
)
from trtvideo.benchmarking.lifecycle import load_frame_markers
from trtvideo.benchmarking.validation import OutputContract, validate_output

PRODUCT_NAME = "TheAnimeScripter"
CommandFactory = Callable[[Path, int], CommandSpec]
_IMPLEMENTATION_PARAMETER_KEYS = (
    "execution_profile",
    "decode_method",
    "writer",
    "cuda_graph",
    "batch_size",
    "full_frame",
    "tiling",
    "bitrate_validation",
    "encoder",
)


def suite_implementation_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the TAS execution contract independently of VapourSynth fields."""
    return {key: parameters[key] for key in _IMPLEMENTATION_PARAMETER_KEYS}


@dataclass(frozen=True)
class TasProcessPaths:
    """Files exchanged with the TAS subprocess adapter for one invocation."""

    encoder_contract: Path
    lifecycle: Path
    runtime_evidence: Path

    @classmethod
    def for_output(cls, output: Path) -> TasProcessPaths:
        return cls(
            encoder_contract=output.parent / "encoder.contract.json",
            lifecycle=output.with_suffix(".lifecycle.json"),
            runtime_evidence=output.with_suffix(".runtime-evidence.json"),
        )


@dataclass(frozen=True)
class TasSuiteConfig:
    """Composition of one TAS workload with the common repeat and media contracts."""

    implementation: dict[str, Any]
    workload_id: str
    variant: str
    output_dir: Path
    frames: int
    warmup_frames: int
    output_contract: dict[str, Any]
    benchmark_contract: dict[str, Any]
    assets: dict[str, Path]
    policy: SuitePolicy
    sample_interval_ms: int
    gpu_id: int
    profile: TasExecutionProfile
    implementation_parameters: dict[str, Any]
    runtime_versions: dict[str, str]
    command: CommandFactory
    keep_outputs: bool = False


def load_runtime_evidence(
    path: Path,
    *,
    profile: TasExecutionProfile,
    frames: int,
    engine_sha256: str,
    onnx_sha256: str,
    source_revision: str,
    adapter_sha256: str,
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Require actual I/O backends and reuse of the prebuilt canonical engine.

    Adapter schema v1 reports effective (not requested) decode_method/writer,
    cuda_graph, engine_reused, processed_frames, engine_sha256, onnx_sha256,
    source_revision, adapter_sha256 and a runtime version object. A valid
    document has status='valid' and an empty errors list.
    """
    evidence = load_json(path)
    if type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
        raise CompetitorError("Unsupported TAS runtime evidence schema")
    if evidence.get("status") != "valid" or evidence.get("errors") != []:
        raise CompetitorError(f"TAS runtime evidence is not valid: {evidence.get('errors')!r}")
    expected: dict[str, str | bool | int] = {
        "decode_method": profile.decode_method,
        "writer": profile.writer,
        "cuda_graph": True,
        "engine_reused": True,
        "processed_frames": frames,
        "engine_sha256": engine_sha256,
        "onnx_sha256": onnx_sha256,
        "source_revision": source_revision,
        "adapter_sha256": adapter_sha256,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if type(evidence.get(key)) is not type(value) or evidence.get(key) != value
    ]
    if mismatches:
        raise CompetitorError(
            "TAS runtime evidence differs from the contract: " + ", ".join(mismatches)
        )
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict):
        raise CompetitorError("TAS runtime evidence has no runtime version object")
    for name in ("python", "torch", "tensorrt", "nelux", "ffmpeg"):
        version = runtime.get(name)
        if not isinstance(version, str) or version.strip().lower() in {"", "unknown"}:
            raise CompetitorError(f"TAS runtime evidence has no {name} version")
    for name, version in runtime_versions.items():
        if runtime.get(name) != version:
            raise CompetitorError(f"TAS runtime evidence {name} differs from the engine build")
    return evidence


def run_tas_command(spec: CommandSpec, stdout_path: Path, stderr_path: Path) -> ProcessResult:
    """Time the adapter's full lifetime; it must wait for every upstream subprocess."""
    if len(spec) != 1 or not spec[0]:
        raise CompetitorError("TAS requires one argv-only subprocess adapter command")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        started_ns = time.perf_counter_ns()
        try:
            process = subprocess.Popen(spec[0], stdout=stdout, stderr=stderr)
            returncode = process.wait()
        except OSError as exc:
            stderr.write(f"Failed to start TAS adapter: {exc}\n".encode())
            returncode = 127
        finished_ns = time.perf_counter_ns()
    return ProcessResult(
        returncode=returncode,
        process_started_ns=started_ns,
        process_finished_ns=finished_ns,
    )


def _contract(config: TasSuiteConfig, frames: int, *, bitrate: bool) -> OutputContract:
    values = dict(config.output_contract)
    values["frames"] = frames
    if not bitrate:
        values["target_bitrate_mbps"] = None
    return OutputContract(**values)


def _run_one(
    config: TasSuiteConfig,
    *,
    run_index: int,
    sampler: Any,
    environment: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    expected_adapter_sha256: str,
    root: Path,
) -> dict[str, Any]:
    paths = VideoRunPaths.create(config.output_dir, run_index)
    warmup_paths = TasProcessPaths.for_output(paths.warmup_output)
    measured_paths = TasProcessPaths.for_output(paths.measured_output)
    write_json(measured_paths.encoder_contract, config.implementation_parameters["encoder"])
    warmup_spec = config.command(paths.warmup_output, config.warmup_frames)
    measured_spec = config.command(paths.measured_output, config.frames)

    def validate(path: Path, contract: OutputContract) -> dict[str, Any]:
        # Shared core invokes this after the timed process and sampler have stopped.
        result = validate_output(path, contract)
        errors = list(result.get("errors", []))
        try:
            result["runtime_evidence"] = load_runtime_evidence(
                TasProcessPaths.for_output(path).runtime_evidence,
                profile=config.profile,
                frames=contract.frames,
                engine_sha256=assets["engine"]["sha256"],
                onnx_sha256=assets["onnx"]["sha256"],
                source_revision=config.implementation["source_revision"],
                adapter_sha256=expected_adapter_sha256,
                runtime_versions=config.runtime_versions,
            )
        except CompetitorError as exc:
            errors.append(f"Runtime evidence: {exc}")
        return {**result, "valid": bool(result.get("valid")) and not errors, "errors": errors}

    return run_video_measurement(
        VideoRunSpec(
            run_index=run_index,
            frames=config.frames,
            warmup_frames=config.warmup_frames,
            keep_outputs=config.keep_outputs,
            max_compute_processes=config.profile.max_compute_processes,
            max_graphics_processes=0,
            require_reproducible_environment=True,
            manifest_fields={
                "product": PRODUCT_NAME,
                "backend": "tas",
                "workload_id": config.workload_id,
                "benchmark_contract_version": config.benchmark_contract["contract_version"],
                "variant": config.variant,
                "implementation": config.implementation,
                "parameters": {
                    "frames": config.frames,
                    "warmup_frames": config.warmup_frames,
                    "gpu_id": config.gpu_id,
                    "nvml_sample_interval_ms": config.sample_interval_ms,
                    "max_compute_processes": config.profile.max_compute_processes,
                    "max_graphics_processes": 0,
                    **config.implementation_parameters,
                },
                "assets": assets,
                "environment": environment,
            },
            warmup=ProcessInvocation(
                command=[sanitize_command(command, root) for command in warmup_spec],
                execute=lambda stdout, stderr: run_tas_command(warmup_spec, stdout, stderr),
            ),
            measured=ProcessInvocation(
                command=[sanitize_command(command, root) for command in measured_spec],
                execute=lambda stdout, stderr: run_tas_command(measured_spec, stdout, stderr),
            ),
            warmup_contract=_contract(config, config.warmup_frames, bitrate=False),
            measured_contract=_contract(config, config.frames, bitrate=True),
            lifecycle_reader=lambda _result: load_frame_markers(measured_paths.lifecycle),
            extra_artifacts={
                "encoder_contract": measured_paths.encoder_contract,
                "warmup_lifecycle": warmup_paths.lifecycle,
                "warmup_runtime_evidence": warmup_paths.runtime_evidence,
                "lifecycle": measured_paths.lifecycle,
                "runtime_evidence": measured_paths.runtime_evidence,
            },
        ),
        paths=paths,
        sampler=sampler,
        root=root,
        validate=validate,
    )


def run_tas_suite(
    config: TasSuiteConfig, *, root: Path | None = None
) -> tuple[dict[str, Any], int]:
    """Run one canonical TAS suite using the shared CPU/NVML/media acceptance core."""
    root = (root or Path.cwd()).resolve()
    if {
        key: config.implementation_parameters.get(key) for key in config.profile.as_parameters()
    } != (config.profile.as_parameters()):
        raise CompetitorError("TAS suite parameters differ from its execution profile")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if any(config.output_dir.iterdir()):
        raise CompetitorError(
            "Benchmark output directory is not empty; remove it or choose "
            f"a unique path: {config.output_dir}"
        )
    try:
        assets = {name: asset_record(name, path, root) for name, path in config.assets.items()}
    except FileNotFoundError as exc:
        raise CompetitorError(str(exc)) from exc
    expected_adapter_sha256 = adapter_sha256()

    def executor_factory(sampler: Any, gpu: dict[str, Any]) -> Any:
        environment = collect_environment(gpu)
        environment.setdefault("software", {}).update(config.runtime_versions)
        environment["image"] = collect_image_identity(
            default_reference=config.implementation["image"]
        )
        environment["implementation"] = config.implementation

        def execute_run(run_index: int) -> dict[str, Any]:
            return _run_one(
                config,
                run_index=run_index,
                sampler=sampler,
                environment=environment,
                assets=assets,
                expected_adapter_sha256=expected_adapter_sha256,
                root=root,
            )

        return execute_run

    return run_video_suite(
        VideoSuiteSpec(
            output_dir=config.output_dir,
            policy=config.policy,
            label=PRODUCT_NAME,
            frames=config.frames,
            warmup_frames=config.warmup_frames,
            sample_interval_ms=config.sample_interval_ms,
            gpu_id=config.gpu_id,
            benchmark_contract=config.benchmark_contract,
            parameter_fields={
                "max_compute_processes": config.profile.max_compute_processes,
                "max_graphics_processes": 0,
                **config.implementation_parameters,
            },
            summary_fields={
                "document_type": "benchmark-result",
                "product": PRODUCT_NAME,
                "backend": "tas",
                "workload_id": config.workload_id,
                "benchmark_contract_version": config.benchmark_contract["contract_version"],
                "variant": config.variant,
                "implementation": config.implementation,
            },
        ),
        executor_factory,
    )
