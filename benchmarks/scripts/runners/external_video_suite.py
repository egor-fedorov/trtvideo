"""External process benchmark suite for full-video implementations."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from benchmarks.scripts.runners.common import CommandSpec, CompetitorError
from benchmarks.scripts.runtime.environment import (
    collect_environment,
    sanitize_command,
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
from trtvideo.benchmarking.lifecycle import (
    FrameLifecycleMarkers,
    LifecycleTimingError,
)
from trtvideo.benchmarking.validation import OutputContract

CommandFactory = Callable[[Path, int], CommandSpec]
_VSPipe_FRAME_PATTERN = re.compile(rb"Frame:\s*(\d+)/(\d+)")


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


@dataclass(frozen=True)
class CommandRunResult(ProcessResult):
    """Exit status and monotonic boundaries of an argv-only command pipeline."""

    first_frame_completed_ns: int | None = None
    producer_finished_ns: int | None = None


class _VspipeProgressObserver:
    """Capture the first completed frame from vspipe's native progress stream."""

    def __init__(self) -> None:
        self.first_frame_completed_ns: int | None = None
        self._tail = b""

    def feed(self, chunk: bytes) -> None:
        payload = self._tail + chunk
        if self.first_frame_completed_ns is None:
            match = _VSPipe_FRAME_PATTERN.search(payload)
            if match is not None and int(match.group(1)) >= 1:
                self.first_frame_completed_ns = time.perf_counter_ns()
        self._tail = payload[-128:]


def _copy_progress_stream(
    stream: Any,
    sink: BinaryIO,
    observer: _VspipeProgressObserver,
) -> None:
    while chunk := stream.read1(4096):
        sink.write(chunk)
        observer.feed(chunk)


def _sanitize_spec(spec: CommandSpec, root: Path) -> list[list[str]]:
    return [sanitize_command(command, root) for command in spec]


def run_command_spec(
    spec: CommandSpec,
    stdout_path: Path,
    stderr_path: Path,
    *,
    observe_vspipe_progress: bool = False,
) -> CommandRunResult:
    """Run an argv-only command pipeline without invoking a shell."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    previous_stdout = None
    started_ns = time.perf_counter_ns()
    observer = _VspipeProgressObserver()
    progress_stream = None
    try:
        with (
            stdout_path.open("wb") as stdout,
            stderr_path.open("wb", buffering=0) as stderr,
        ):
            for index, command in enumerate(spec):
                is_last = index == len(spec) - 1
                capture_progress = observe_vspipe_progress and index == 0
                process = subprocess.Popen(
                    command,
                    stdin=previous_stdout,
                    stdout=stdout if is_last else subprocess.PIPE,
                    stderr=subprocess.PIPE if capture_progress else stderr,
                )
                if previous_stdout is not None:
                    previous_stdout.close()
                previous_stdout = process.stdout
                processes.append(process)
                if capture_progress:
                    progress_stream = process.stderr

            progress_thread = None
            if progress_stream is not None:
                progress_thread = threading.Thread(
                    target=_copy_progress_stream,
                    args=(progress_stream, stderr, observer),
                    daemon=True,
                )
                progress_thread.start()

            returncodes: list[int | None] = [None] * len(processes)
            finished_ns: list[int | None] = [None] * len(processes)

            def wait_for_process(index: int, process: subprocess.Popen[bytes]) -> None:
                returncodes[index] = process.wait()
                finished_ns[index] = time.perf_counter_ns()

            waiters = [
                threading.Thread(
                    target=wait_for_process,
                    args=(index, process),
                    daemon=True,
                )
                for index, process in enumerate(processes)
            ]
            for waiter in waiters:
                waiter.start()
            for waiter in waiters:
                waiter.join()
            if progress_thread is not None:
                progress_thread.join()
    except OSError as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        stderr_path.write_text(f"Failed to start pipeline: {exc}\n", encoding="utf-8")
        return CommandRunResult(
            returncode=127,
            process_started_ns=started_ns,
            process_finished_ns=time.perf_counter_ns(),
        )

    completed_codes = [code for code in returncodes if code is not None]
    completed_times = [value for value in finished_ns if value is not None]
    return CommandRunResult(
        returncode=next((code for code in completed_codes if code != 0), 0),
        process_started_ns=started_ns,
        process_finished_ns=max(completed_times, default=time.perf_counter_ns()),
        first_frame_completed_ns=observer.first_frame_completed_ns,
        producer_finished_ns=finished_ns[0] if finished_ns else None,
    )


def _environment(config: ExternalVideoSuiteConfig, gpu: dict[str, Any]) -> dict[str, Any]:
    environment = collect_environment(gpu)
    environment["image"] = {
        "reference": os.environ.get(
            "TRTVIDEO_IMAGE_REF", config.implementation.metadata["image"]
        ),
        "id": os.environ.get("TRTVIDEO_IMAGE_ID", "unknown"),
        "base_reference": os.environ.get("TRTVIDEO_BASE_IMAGE", "unknown"),
        "repository_revision": os.environ.get("TRTVIDEO_BUILD_REVISION", "unknown"),
        "source_dirty": os.environ.get("TRTVIDEO_BUILD_DIRTY", "unknown"),
    }
    environment["implementation"] = config.implementation.metadata
    return environment


def _contract(config: ExternalVideoSuiteConfig, frames: int, *, bitrate: bool) -> OutputContract:
    values = dict(config.workload.output_contract)
    values["frames"] = frames
    if not bitrate:
        values["target_bitrate_mbps"] = None
    return OutputContract(**values)


def _run_one(
    config: ExternalVideoSuiteConfig,
    *,
    run_index: int,
    sampler: Any,
    environment: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    workload = config.workload
    implementation = config.implementation
    paths = VideoRunPaths.create(workload.output_dir, run_index)
    warmup_spec = config.warmup_command(paths.warmup_output, workload.warmup_frames)
    measured_spec = config.measured_command(paths.measured_output, workload.frames)

    def lifecycle_reader(result: ProcessResult) -> FrameLifecycleMarkers:
        if not isinstance(result, CommandRunResult):
            raise LifecycleTimingError("vspipe returned an unexpected process result")
        if (
            result.first_frame_completed_ns is None
            or result.producer_finished_ns is None
        ):
            raise LifecycleTimingError(
                "vspipe did not expose the required frame boundaries"
            )
        return FrameLifecycleMarkers(
            first_frame_completed_ns=result.first_frame_completed_ns,
            last_frame_completed_ns=result.producer_finished_ns,
            processed_frames=workload.frames,
            instrumentation="vspipe-progress-and-producer-exit",
        )

    return run_video_measurement(
        VideoRunSpec(
            run_index=run_index,
            frames=workload.frames,
            warmup_frames=workload.warmup_frames,
            keep_outputs=config.keep_outputs,
            max_compute_processes=implementation.max_compute_processes,
            max_graphics_processes=implementation.max_graphics_processes,
            require_reproducible_environment=True,
            manifest_fields={
                "product": implementation.product,
                "backend": implementation.backend,
                "comparison_class": implementation.comparison_class,
                "workload_id": workload.workload_id,
                "benchmark_contract_version": workload.benchmark_contract[
                    "contract_version"
                ],
                "variant": workload.variant,
                "implementation": implementation.metadata,
                "parameters": {
                    "frames": workload.frames,
                    "warmup_frames": workload.warmup_frames,
                    "gpu_id": config.gpu_id,
                    "nvml_sample_interval_ms": config.sample_interval_ms,
                    "max_compute_processes": implementation.max_compute_processes,
                    "max_graphics_processes": implementation.max_graphics_processes,
                    **config.implementation_parameters,
                },
                "assets": assets,
                "environment": environment,
            },
            warmup=ProcessInvocation(
                command=_sanitize_spec(warmup_spec, root),
                execute=lambda stdout, stderr: run_command_spec(
                    warmup_spec,
                    stdout,
                    stderr,
                ),
            ),
            measured=ProcessInvocation(
                command=_sanitize_spec(measured_spec, root),
                execute=lambda stdout, stderr: run_command_spec(
                    measured_spec,
                    stdout,
                    stderr,
                    observe_vspipe_progress=True,
                ),
            ),
            warmup_contract=_contract(
                config,
                workload.warmup_frames,
                bitrate=False,
            ),
            measured_contract=_contract(
                config,
                workload.frames,
                bitrate=True,
            ),
            lifecycle_reader=lifecycle_reader,
        ),
        paths=paths,
        sampler=sampler,
        root=root,
    )


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
    if any(workload.output_dir.iterdir()):
        raise CompetitorError(
            "Benchmark output directory is not empty; remove it or choose "
            f"a unique path: {workload.output_dir}"
        )
    try:
        assets = {
            name: asset_record(name, path, root)
            for name, path in workload.assets.items()
        }
    except FileNotFoundError as exc:
        raise CompetitorError(str(exc)) from exc

    def executor_factory(sampler: Any, gpu: dict[str, Any]) -> Any:
        environment = _environment(config, gpu)

        def execute_run(run_index: int) -> dict[str, Any]:
            return _run_one(
                config,
                run_index=run_index,
                sampler=sampler,
                environment=environment,
                assets=assets,
                root=root,
            )

        return execute_run

    return run_video_suite(
        VideoSuiteSpec(
            output_dir=workload.output_dir,
            policy=config.policy,
            label=implementation.product,
            frames=workload.frames,
            warmup_frames=workload.warmup_frames,
            sample_interval_ms=config.sample_interval_ms,
            gpu_id=config.gpu_id,
            benchmark_contract=workload.benchmark_contract,
            parameter_fields={
                "max_compute_processes": implementation.max_compute_processes,
                "max_graphics_processes": implementation.max_graphics_processes,
                **config.implementation_parameters,
            },
            summary_fields={
                "document_type": "benchmark-result",
                "product": implementation.product,
                "backend": implementation.backend,
                "comparison_class": implementation.comparison_class,
                "workload_id": workload.workload_id,
                "benchmark_contract_version": workload.benchmark_contract[
                    "contract_version"
                ],
                "variant": workload.variant,
                "implementation": implementation.metadata,
            },
        ),
        executor_factory,
    )
