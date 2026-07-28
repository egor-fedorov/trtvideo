"""Process-level benchmark orchestration without per-frame instrumentation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.scripts.runtime.cpu import snapshot_child_cpu, summarize_child_cpu
from benchmarks.scripts.runtime.environment import (
    collect_environment,
    environment_errors,
    relative_artifact_path,
    sanitize_command,
    sha256_file,
    write_json,
)
from benchmarks.scripts.runtime.nvml import NvmlSampler, summarize_samples, write_samples
from benchmarks.scripts.runtime.suite import (
    SuitePolicy,
    SuiteRunner,
    canonical_suite_errors,
    report_publishability_errors,
    suite_publishability_errors,
)
from trtvideo.benchmarking.lifecycle import (
    LifecycleTimingError,
    load_frame_markers,
    median_detailed_phase_intervals,
    summarize_lifecycle,
)
from trtvideo.benchmarking.validation import OutputContract, validate_output
from trtvideo.video.fps import gop_size_for_one_second
from trtvideo.video.info import get_video_info
from trtvideo.video.nvenc import NvencCbrContract

PRODUCT_NAME = "trtvideo"


class BenchmarkError(RuntimeError):
    """Raised for invalid benchmark configuration or assets."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by every run in one suite."""

    engine: Path
    input_path: Path
    output_dir: Path
    backend: str = "nvcodec"
    gpu_id: int = 0
    frames: int = 1000
    warmup_frames: int = 100
    initial_runs: int = 3
    extra_runs: int = 2
    spread_threshold: float = 0.05
    idle_seconds: float = 10.0
    sample_interval_ms: int = 100
    bitrate_mbps: float | None = None
    crf: int = 18
    cuda_graph: bool = False
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
    if config.backend not in {"ffmpeg", "nvcodec"}:
        raise BenchmarkError(f"Unsupported backend: {config.backend}")
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
    if config.backend == "nvcodec" and (
        config.bitrate_mbps is None or config.bitrate_mbps <= 0
    ):
        raise BenchmarkError("nvcodec benchmark requires explicit positive --bitrate-mbps")
    if config.backend == "ffmpeg" and not 0 <= config.crf <= 51:
        raise BenchmarkError("--crf must be in the range 0..51")
    if (config.workload_manifest is None) != (config.variant is None):
        raise BenchmarkError("--workload-manifest and --variant must be used together")


def load_engine_contract(engine: Path) -> tuple[dict[str, Any], Path]:
    """Load and verify the sidecar emitted by build-engine."""
    sidecar_path = Path(f"{engine}.json")
    if not sidecar_path.is_file():
        raise BenchmarkError(f"Engine sidecar not found: {sidecar_path}")
    sidecar = load_json(sidecar_path)
    expected_hash = sidecar.get("engine_sha256")
    actual_hash = sha256_file(engine)
    if expected_hash != actual_hash:
        raise BenchmarkError(
            f"Engine SHA256 does not match sidecar: expected {expected_hash}, got {actual_hash}"
        )
    for tensor_name in ("input", "output"):
        shape = sidecar.get(tensor_name, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 4 or not all(
            isinstance(value, int) and value > 0 for value in shape
        ):
            raise BenchmarkError(f"Engine sidecar has invalid static {tensor_name} shape")
    return sidecar, sidecar_path


def _asset_record(kind: str, path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"Required {kind} asset not found: {path}")
    return {
        "kind": kind,
        "path": relative_artifact_path(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


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
    assets = {
        "input": _asset_record("input", config.input_path, root),
        "engine": _asset_record("engine", config.engine, root),
        "engine_manifest": _asset_record("engine_manifest", sidecar_path, root),
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
    assets["weights"] = _asset_record("weights", weights_path, root)
    assets["onnx"] = _asset_record("onnx", onnx_path, root)
    assets["workload_manifest"] = _asset_record(
        "workload_manifest", config.workload_manifest, root
    )
    assets["asset_lock"] = _asset_record("asset_lock", lock_path, root)
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
        "--backend",
        config.backend,
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
    ]
    if config.backend == "nvcodec":
        command.extend(["--codec", "h264", "--bitrate-mbps", str(config.bitrate_mbps)])
    else:
        command.extend(["--crf", str(config.crf)])
    if config.cuda_graph:
        command.append("--cuda-graph")
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
    info = get_video_info(str(config.input_path))
    input_shape = sidecar["input"]["shape"]
    output_shape = sidecar["output"]["shape"]
    if [info.height, info.width] != [input_shape[2], input_shape[3]]:
        raise BenchmarkError(
            f"Input video {info.width}x{info.height} does not match engine "
            f"{input_shape[3]}x{input_shape[2]}"
        )
    gop_frames = (
        gop_size_for_one_second(info.fps_str) if config.backend == "nvcodec" else None
    )
    return OutputContract(
        width=output_shape[3],
        height=output_shape[2],
        fps=info.fps_str,
        frames=frames,
        has_b_frames=0 if config.backend == "nvcodec" else None,
        gop_frames=gop_frames,
        target_bitrate_mbps=(
            config.bitrate_mbps if config.backend == "nvcodec" and enforce_bitrate else None
        ),
        require_monotonic_pts=config.backend == "nvcodec",
    )


def run_child(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    """Run one child process while keeping terminal output out of timed measurements."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            return process.wait()
    except OSError as exc:
        stderr_path.write_text(f"Failed to start child process: {exc}\n", encoding="utf-8")
        return 127


def _cleanup_valid_output(path: Path, keep_outputs: bool) -> None:
    if not keep_outputs and path.exists():
        path.unlink()


def _hash_if_present(path: Path, kind: str, root: Path) -> dict[str, Any] | None:
    return _asset_record(kind, path, root) if path.is_file() else None


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
    sampler: NvmlSampler,
    root: Path,
    validate: Callable[[Path, OutputContract], dict[str, Any]] = validate_output,
) -> dict[str, Any]:
    """Run one discarded warmup and one externally measured process."""
    run_dir = config.output_dir / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = run_dir / "warmup.mp4"
    measured_output = run_dir / "output.mp4"
    warmup_stdout = run_dir / "warmup.stdout.log"
    warmup_stderr = run_dir / "warmup.stderr.log"
    measured_stdout = run_dir / "measured.stdout.log"
    measured_stderr = run_dir / "measured.stderr.log"
    samples_path = run_dir / "nvml.samples.jsonl"
    lifecycle_path = run_dir / "lifecycle.json"
    manifest_path = run_dir / "manifest.json"

    warmup_command = build_upscale_command(
        config,
        output_path=warmup_output,
        frame_count=config.warmup_frames,
    )
    measured_command = build_upscale_command(
        config,
        output_path=measured_output,
        frame_count=config.frames,
        lifecycle_path=lifecycle_path,
    )
    lifecycle_path.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_index": run_index,
        "product": PRODUCT_NAME,
        "backend": config.backend,
        "workload_id": workload_id,
        "benchmark_contract_version": benchmark_contract_version,
        "variant": config.variant,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "frames": config.frames,
            "warmup_frames": config.warmup_frames,
            "gpu_id": config.gpu_id,
            "bitrate_mbps": config.bitrate_mbps,
            "crf": config.crf if config.backend == "ffmpeg" else None,
            "cuda_graph": config.cuda_graph,
            "bitrate_validation": config.validate_bitrate,
            "nvml_sample_interval_ms": config.sample_interval_ms,
            "encoder": encoder_parameters,
        },
        "commands": {
            "warmup": sanitize_command(warmup_command, root),
            "measured": sanitize_command(measured_command, root),
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
            "lifecycle": relative_artifact_path(lifecycle_path, root),
        },
        "errors": [],
    }

    warmup_returncode = run_child(warmup_command, warmup_stdout, warmup_stderr)
    warmup_validation = (
        validate(
            warmup_output,
            output_contract(
                config,
                sidecar,
                frames=config.warmup_frames,
                enforce_bitrate=False,
            ),
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
    _cleanup_valid_output(warmup_output, config.keep_outputs)

    cpu_before = snapshot_child_cpu()
    sampler.start(time.perf_counter())
    start_time_ns = time.perf_counter_ns()
    try:
        measured_returncode = run_child(measured_command, measured_stdout, measured_stderr)
    finally:
        end_time_ns = time.perf_counter_ns()
        cpu_after = snapshot_child_cpu()
        samples = sampler.samples_relative_to(
            sampler.stop(),
            start_time_ns / 1_000_000_000,
        )
    wall_time_sec = (end_time_ns - start_time_ns) / 1_000_000_000
    cpu_summary = summarize_child_cpu(
        cpu_before,
        cpu_after,
        wall_time_sec=wall_time_sec,
    ).as_dict()
    write_samples(samples_path, samples)
    nvml_summary = summarize_samples(
        samples,
        wall_time_sec=wall_time_sec,
        frames=config.frames,
        max_compute_processes=1,
        max_graphics_processes=0,
    )
    measured_validation = (
        validate(
            measured_output,
            output_contract(
                config,
                sidecar,
                frames=config.frames,
                enforce_bitrate=config.validate_bitrate,
            ),
        )
        if measured_output.is_file()
        else {"valid": False, "errors": ["Measured output was not created"]}
    )
    output_asset = _hash_if_present(measured_output, "output", root)
    lifecycle_summary = None
    lifecycle_error = None
    try:
        lifecycle_summary = summarize_lifecycle(
            process_started_ns=start_time_ns,
            process_finished_ns=end_time_ns,
            markers=load_frame_markers(lifecycle_path),
            expected_frames=config.frames,
        )
    except LifecycleTimingError as exc:
        lifecycle_error = str(exc)
    metrics = {
        "wall_time_sec": wall_time_sec,
        "end_to_end_fps": config.frames / wall_time_sec if wall_time_sec > 0 else None,
        "processed_frames": config.frames,
        "cpu": cpu_summary,
        "lifecycle": lifecycle_summary,
        "nvml": nvml_summary,
    }
    manifest["measured"] = {
        "returncode": measured_returncode,
        "metrics": metrics,
        "validation": measured_validation,
        "output": output_asset,
    }
    if measured_returncode != 0:
        manifest["errors"].append(
            f"Measured process exited with code {measured_returncode}"
        )
    if lifecycle_error is not None:
        manifest["errors"].append(f"Lifecycle timing: {lifecycle_error}")
    if not measured_validation.get("valid"):
        manifest["errors"].extend(measured_validation.get("errors", []))
    if not nvml_summary.get("valid"):
        manifest["errors"].extend(nvml_summary.get("errors", []))
    reproducibility_errors = environment_errors(environment)
    manifest["reproducibility"] = {
        "publishable": not reproducibility_errors,
        "errors": reproducibility_errors,
    }
    if workload_id is not None:
        manifest["errors"].extend(reproducibility_errors)
    manifest["status"] = "valid" if not manifest["errors"] else "invalid"
    write_json(manifest_path, manifest)
    if manifest["status"] == "valid":
        _cleanup_valid_output(measured_output, config.keep_outputs)
    return manifest


def _end_to_end_fps(manifest: dict[str, Any]) -> float:
    value = manifest.get("measured", {}).get("metrics", {}).get("end_to_end_fps")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkError("Valid run manifest has no end_to_end_fps metric")
    return float(value)


def _video_power_limit(manifest: dict[str, Any]) -> float | None:
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
        raise BenchmarkError("Run manifest has an invalid GPU power limit")
    return float(value)


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
        load_json(config.workload_manifest)
        if config.workload_manifest is not None
        else None
    )
    benchmark_contract = (
        workload_manifest.get("benchmark")
        if isinstance(workload_manifest, dict)
        else None
    )
    benchmark_contract_version = (
        int(benchmark_contract["contract_version"])
        if isinstance(benchmark_contract, dict)
        else None
    )
    sampler = NvmlSampler(config.gpu_id, config.sample_interval_ms)
    gpu = sampler.initialize()
    environment = collect_environment(gpu)
    encoder_parameters = None
    if config.backend == "nvcodec":
        assert config.bitrate_mbps is not None
        info = get_video_info(str(config.input_path))
        encoder_parameters = NvencCbrContract(
            bitrate_bps=int(config.bitrate_mbps * 1_000_000),
            gop_frames=gop_size_for_one_second(info.fps_str),
        ).as_dict()
    summary_path = config.output_dir / "suite.json"
    policy = SuitePolicy(
        initial_runs=config.initial_runs,
        extra_runs=config.extra_runs,
        spread_threshold=config.spread_threshold,
        idle_seconds=config.idle_seconds,
    )
    suite_runner = SuiteRunner(
        policy,
        label=PRODUCT_NAME,
        frames=config.frames,
        metric_reader=_end_to_end_fps,
        power_limit_reader=_video_power_limit,
    )

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

    try:
        suite_result = suite_runner.execute(execute_run)
    finally:
        sampler.shutdown()

    run_manifests = list(suite_result.runs)
    statistics_report = suite_result.statistics
    lifecycle_summaries = [
        run.get("measured", {}).get("metrics", {}).get("lifecycle")
        for run in run_manifests
    ]
    statistics_report["median_lifecycle_intervals_sec"] = (
        median_detailed_phase_intervals(lifecycle_summaries)
        if lifecycle_summaries
        and all(isinstance(summary, dict) for summary in lifecycle_summaries)
        else {}
    )
    spread = statistics_report["relative_spread"]
    suite_errors = list(suite_result.errors)
    status = suite_result.status
    parameters = {
        "frames": config.frames,
        "warmup_frames": config.warmup_frames,
        "initial_runs": config.initial_runs,
        "extra_runs_on_spread": config.extra_runs,
        "spread_threshold": config.spread_threshold,
        "idle_seconds": config.idle_seconds,
        "nvml_sample_interval_ms": config.sample_interval_ms,
        "bitrate_mbps": config.bitrate_mbps,
        "cuda_graph": config.cuda_graph,
        "encoder": encoder_parameters,
    }
    canonical_errors = canonical_suite_errors(
        parameters,
        benchmark_contract if isinstance(benchmark_contract, dict) else None,
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
        "status": status,
        "scope": "acceptance",
        "publishable": not publishability_errors,
        "publishability": {
            "canonical_contract": not canonical_errors,
            "errors": publishability_errors,
        },
        "product": PRODUCT_NAME,
        "backend": config.backend,
        "workload_id": workload_id,
        "benchmark_contract_version": benchmark_contract_version,
        "variant": config.variant,
        "parameters": parameters,
        "statistics": statistics_report,
        "errors": suite_errors,
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
        f"Benchmark suite {status}: median={statistics_report['median_fps']!r} FPS, "
        f"spread={spread!r}",
        file=sys.stderr,
    )
    report_publishability_errors(
        publishability_errors,
        acceptance_only=True,
    )
    return summary, 0 if status == "valid" else 2


def write_summary_target(path: str | None, summary: dict[str, Any]) -> None:
    """Optionally mirror the canonical suite summary to a path or stdout."""
    if path is None:
        return
    if path == "-":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    write_json(Path(path), summary)
