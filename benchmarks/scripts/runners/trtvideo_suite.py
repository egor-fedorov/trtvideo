"""Process-level benchmark suite for the trtvideo product."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.engine import load_engine_contract
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
from trtvideo.benchmarking.lifecycle import load_frame_markers
from trtvideo.benchmarking.validation import OutputContract
from trtvideo.video.nvcodec.encoder import NvencCbrContract, gop_size_for_one_second
from trtvideo.video.probe import probe_video

PRODUCT_NAME = "trtvideo"


class BenchmarkError(RuntimeError):
    """Raised for invalid benchmark configuration or assets."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by every run in one suite."""

    engine: Path
    input_path: Path
    output_dir: Path
    gpu_id: int = 0
    frames: int = 1000
    warmup_frames: int = 100
    initial_runs: int = 3
    extra_runs: int = 2
    spread_threshold: float = 0.05
    idle_seconds: float = 10.0
    sample_interval_ms: int = 100
    bitrate_mbps: float | None = None
    keep_outputs: bool = False
    validate_bitrate: bool = True
    workload_manifest: Path | None = None
    variant: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with a benchmark-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected a JSON object in {path}")
    return value


def validate_config(config: BenchmarkConfig) -> None:
    """Reject configurations that cannot produce comparable measurements."""
    if not config.engine.is_file():
        raise BenchmarkError(f"Engine not found: {config.engine}")
    if not config.input_path.is_file():
        raise BenchmarkError(f"Input video not found: {config.input_path}")
    if config.frames <= 0 or config.warmup_frames <= 0:
        raise BenchmarkError("--frames and --warmup-frames must be greater than zero")
    if config.initial_runs <= 0 or config.extra_runs < 0:
        raise BenchmarkError("--runs must be positive and --extra-runs cannot be negative")
    if not 0 <= config.spread_threshold < 1:
        raise BenchmarkError("--spread-threshold must be in the range [0, 1)")
    if config.idle_seconds < 0:
        raise BenchmarkError("--idle-seconds cannot be negative")
    if config.sample_interval_ms <= 0:
        raise BenchmarkError("--nvml-sample-ms must be greater than zero")
    if config.bitrate_mbps is None or config.bitrate_mbps <= 0:
        raise BenchmarkError("Benchmark requires explicit positive --bitrate-mbps")
    if (config.workload_manifest is None) != (config.variant is None):
        raise BenchmarkError("--workload-manifest and --variant must be used together")


def _find_locked_asset(lock: dict[str, Any], *, kind: str, variant: str) -> dict[str, Any]:
    for asset in lock.get("assets", []):
        if asset.get("kind") == kind and asset.get("variant") == variant:
            return asset
    raise BenchmarkError(f"Asset lock has no {kind} entry for variant {variant}")


def collect_assets(
    config: BenchmarkConfig,
    sidecar: dict[str, Any],
    sidecar_path: Path,
    root: Path,
) -> tuple[dict[str, Any], str | None]:
    """Hash required assets and verify optional canonical workload metadata."""

    def record(kind: str, path: Path) -> dict[str, Any]:
        try:
            return asset_record(kind, path, root)
        except FileNotFoundError as exc:
            raise BenchmarkError(str(exc)) from exc

    assets = {
        "input": record("input", config.input_path),
        "engine": record("engine", config.engine),
        "engine_manifest": record("engine_manifest", sidecar_path),
    }
    workload_id: str | None = None
    if config.workload_manifest is None or config.variant is None:
        return assets, workload_id

    manifest = load_json(config.workload_manifest)
    workload_id = str(manifest.get("id", "unknown"))
    lock_path = root / str(manifest.get("lock_path", ""))
    lock = load_json(lock_path)
    if lock.get("workload_id") != workload_id:
        raise BenchmarkError("Asset lock workload ID does not match workload manifest")

    input_lock = _find_locked_asset(lock, kind="input_video", variant=config.variant)
    onnx_lock = _find_locked_asset(lock, kind="onnx", variant=config.variant)
    clip_variant = next(
        (
            item
            for item in manifest.get("clip", {}).get("variants", [])
            if item.get("name") == config.variant
        ),
        None,
    )
    if clip_variant is None:
        raise BenchmarkError(f"Workload manifest has no clip variant {config.variant}")
    expected_output = clip_variant.get("benchmark_output", {})
    actual_output_shape = sidecar["output"]["shape"]
    if [expected_output.get("height"), expected_output.get("width")] != [
        actual_output_shape[2],
        actual_output_shape[3],
    ]:
        raise BenchmarkError("Engine output shape does not match canonical benchmark output")
    if config.bitrate_mbps != expected_output.get("bitrate_mbps"):
        raise BenchmarkError("NVENC bitrate does not match canonical benchmark output")
    if assets["input"]["sha256"] != input_lock.get("sha256"):
        raise BenchmarkError("Input video SHA256 does not match canonical asset lock")
    if sidecar.get("model_sha256") != onnx_lock.get("sha256"):
        raise BenchmarkError("Engine ONNX SHA256 does not match canonical asset lock")

    weights_path = root / str(manifest.get("model", {}).get("weights_path", ""))
    onnx_path = root / str(onnx_lock.get("path", ""))
    assets["weights"] = record("weights", weights_path)
    assets["onnx"] = record("onnx", onnx_path)
    assets["workload_manifest"] = record(
        "workload_manifest",
        config.workload_manifest,
    )
    assets["asset_lock"] = record("asset_lock", lock_path)
    expected_weights_hash = manifest.get("model", {}).get("source", {}).get("sha256")
    if assets["weights"]["sha256"] != expected_weights_hash:
        raise BenchmarkError("Weights SHA256 does not match workload manifest")
    if assets["onnx"]["sha256"] != onnx_lock.get("sha256"):
        raise BenchmarkError("ONNX SHA256 does not match canonical asset lock")
    return assets, workload_id


def build_upscale_command(
    config: BenchmarkConfig,
    *,
    output_path: Path,
    frame_count: int,
    lifecycle_path: Path | None = None,
) -> list[str]:
    """Build an unprofiled child command for warmup or measurement."""
    command = [
        "upscale",
        "--engine",
        str(config.engine),
        "--input",
        str(config.input_path),
        "--output",
        str(output_path),
        "--gpu-id",
        str(config.gpu_id),
        "--max-frames",
        str(frame_count),
        "--quiet",
        "--codec",
        "h264",
        "--bitrate-mbps",
        str(config.bitrate_mbps),
    ]
    if lifecycle_path is not None:
        command.extend(["--benchmark-lifecycle-json", str(lifecycle_path)])
    return command


def output_contract(
    config: BenchmarkConfig,
    sidecar: dict[str, Any],
    *,
    frames: int,
    enforce_bitrate: bool,
) -> OutputContract:
    """Derive exact output expectations from input metadata and engine sidecar."""
    info = probe_video(str(config.input_path))
    input_shape = sidecar["input"]["shape"]
    output_shape = sidecar["output"]["shape"]
    if [info.height, info.width] != [input_shape[2], input_shape[3]]:
        raise BenchmarkError(
            f"Input video {info.width}x{info.height} does not match engine "
            f"{input_shape[3]}x{input_shape[2]}"
        )
    gop_frames = gop_size_for_one_second(info.fps_str)
    return OutputContract(
        width=output_shape[3],
        height=output_shape[2],
        fps=info.fps_str,
        frames=frames,
        has_b_frames=0,
        gop_frames=gop_frames,
        target_bitrate_mbps=config.bitrate_mbps if enforce_bitrate else None,
        require_monotonic_pts=True,
    )


def run_child(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> ProcessResult:
    """Run one child process while keeping terminal output out of timed measurements."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.perf_counter_ns()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            returncode = process.wait()
    except OSError as exc:
        stderr_path.write_text(f"Failed to start child process: {exc}\n", encoding="utf-8")
        returncode = 127
    return ProcessResult(
        returncode=returncode,
        process_started_ns=started_ns,
        process_finished_ns=time.perf_counter_ns(),
    )


def run_one(
    config: BenchmarkConfig,
    *,
    run_index: int,
    sidecar: dict[str, Any],
    assets: dict[str, Any],
    workload_id: str | None,
    benchmark_contract_version: int | None,
    environment: dict[str, Any],
    encoder_parameters: dict[str, Any] | None,
    sampler: Any,
    root: Path,
) -> dict[str, Any]:
    """Run one discarded warmup and one externally measured process."""
    paths = VideoRunPaths.create(config.output_dir, run_index)
    lifecycle_path = paths.run_dir / "lifecycle.json"
    warmup_command = build_upscale_command(
        config,
        output_path=paths.warmup_output,
        frame_count=config.warmup_frames,
    )
    measured_command = build_upscale_command(
        config,
        output_path=paths.measured_output,
        frame_count=config.frames,
        lifecycle_path=lifecycle_path,
    )
    lifecycle_path.unlink(missing_ok=True)
    return run_video_measurement(
        VideoRunSpec(
            run_index=run_index,
            frames=config.frames,
            warmup_frames=config.warmup_frames,
            keep_outputs=config.keep_outputs,
            max_compute_processes=1,
            max_graphics_processes=0,
            require_reproducible_environment=workload_id is not None,
            manifest_fields={
                "product": PRODUCT_NAME,
                "workload_id": workload_id,
                "benchmark_contract_version": benchmark_contract_version,
                "variant": config.variant,
                "parameters": {
                    "frames": config.frames,
                    "warmup_frames": config.warmup_frames,
                    "gpu_id": config.gpu_id,
                    "bitrate_mbps": config.bitrate_mbps,
                    "cuda_graph": False,
                    "bitrate_validation": config.validate_bitrate,
                    "nvml_sample_interval_ms": config.sample_interval_ms,
                    "encoder": encoder_parameters,
                },
                "assets": assets,
                "environment": environment,
            },
            warmup=ProcessInvocation(
                command=sanitize_command(warmup_command, root),
                execute=lambda stdout, stderr: run_child(
                    warmup_command,
                    stdout,
                    stderr,
                ),
            ),
            measured=ProcessInvocation(
                command=sanitize_command(measured_command, root),
                execute=lambda stdout, stderr: run_child(
                    measured_command,
                    stdout,
                    stderr,
                ),
            ),
            warmup_contract=output_contract(
                config,
                sidecar,
                frames=config.warmup_frames,
                enforce_bitrate=False,
            ),
            measured_contract=output_contract(
                config,
                sidecar,
                frames=config.frames,
                enforce_bitrate=config.validate_bitrate,
            ),
            lifecycle_reader=lambda _result: load_frame_markers(lifecycle_path),
            extra_artifacts={"lifecycle": lifecycle_path},
        ),
        paths=paths,
        sampler=sampler,
        root=root,
    )


def run_suite(config: BenchmarkConfig, root: Path | None = None) -> tuple[dict[str, Any], int]:
    """Run a complete 3+2 suite and return its machine-readable summary."""
    validate_config(config)
    root = (root or Path.cwd()).resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if any(config.output_dir.iterdir()):
        raise BenchmarkError(
            "Benchmark output directory is not empty; remove it or choose "
            f"a unique path: {config.output_dir}"
        )
    sidecar, sidecar_path = load_engine_contract(config.engine)
    assets, workload_id = collect_assets(config, sidecar, sidecar_path, root)
    workload_manifest = (
        load_json(config.workload_manifest) if config.workload_manifest is not None else None
    )
    benchmark_contract = (
        workload_manifest.get("benchmark") if isinstance(workload_manifest, dict) else None
    )
    benchmark_contract_version = (
        int(benchmark_contract["contract_version"])
        if isinstance(benchmark_contract, dict)
        else None
    )
    assert config.bitrate_mbps is not None
    info = probe_video(str(config.input_path))
    encoder_parameters = NvencCbrContract(
        bitrate_bps=int(config.bitrate_mbps * 1_000_000),
        gop_frames=gop_size_for_one_second(info.fps_str),
    ).as_dict()
    policy = SuitePolicy(
        initial_runs=config.initial_runs,
        extra_runs=config.extra_runs,
        spread_threshold=config.spread_threshold,
        idle_seconds=config.idle_seconds,
    )

    def executor_factory(sampler: Any, gpu: dict[str, Any]) -> Any:
        environment = collect_environment(gpu)

        def execute_run(run_index: int) -> dict[str, Any]:
            return run_one(
                config,
                run_index=run_index,
                sidecar=sidecar,
                assets=assets,
                workload_id=workload_id,
                benchmark_contract_version=benchmark_contract_version,
                environment=environment,
                encoder_parameters=encoder_parameters,
                sampler=sampler,
                root=root,
            )

        return execute_run

    return run_video_suite(
        VideoSuiteSpec(
            output_dir=config.output_dir,
            policy=policy,
            label=PRODUCT_NAME,
            frames=config.frames,
            warmup_frames=config.warmup_frames,
            sample_interval_ms=config.sample_interval_ms,
            gpu_id=config.gpu_id,
            benchmark_contract=(
                benchmark_contract if isinstance(benchmark_contract, dict) else None
            ),
            parameter_fields={
                "bitrate_mbps": config.bitrate_mbps,
                "cuda_graph": False,
                "encoder": encoder_parameters,
            },
            summary_fields={
                "product": PRODUCT_NAME,
                "workload_id": workload_id,
                "benchmark_contract_version": benchmark_contract_version,
                "variant": config.variant,
            },
            include_median_lifecycle=True,
        ),
        executor_factory,
    )
