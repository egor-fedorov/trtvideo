"""Process-level benchmark orchestration without per-frame instrumentation."""

from __future__ import annotations

import json
import statistics
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
from ai_media.benchmarking.validation import OutputContract, validate_output
from ai_media.video.fps import gop_size_for_one_second
from ai_media.video.info import get_video_info


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


def compute_suite_statistics(fps_values: list[float]) -> dict[str, Any]:
    """Aggregate repeat measurements without hiding raw values."""
    if not fps_values:
        return {
            "values_fps": [],
            "median_fps": None,
            "min_fps": None,
            "max_fps": None,
            "relative_spread": None,
        }
    median = statistics.median(fps_values)
    minimum = min(fps_values)
    maximum = max(fps_values)
    return {
        "values_fps": fps_values,
        "median_fps": median,
        "min_fps": minimum,
        "max_fps": maximum,
        "relative_spread": (maximum - minimum) / median if median > 0 else None,
    }


def should_extend_suite(fps_values: list[float], threshold: float) -> bool:
    """Return whether the initial suite is too noisy for the fixed 3-run contract."""
    spread = compute_suite_statistics(fps_values)["relative_spread"]
    return spread is not None and spread > threshold


def report_invalid_run(run: dict[str, Any]) -> None:
    """Print manifest errors when a measured run invalidates its suite."""
    print(f"Benchmark run {run.get('run_index', '?')} invalid:", file=sys.stderr)
    errors = run.get("errors", [])
    if not errors:
        print("  - No detailed error was recorded", file=sys.stderr)
        return
    for error in errors:
        print(f"  - {error}", file=sys.stderr)


def canonical_suite_errors(
    parameters: dict[str, Any],
    benchmark: dict[str, Any] | None,
    *,
    include_warmup_frames: bool,
) -> list[str]:
    """Explain why suite parameters do not match the publication contract."""
    if benchmark is None:
        return ["No canonical workload benchmark contract was provided"]

    expected_keys = {
        "frames": "measured_frames",
        "initial_runs": "initial_runs",
        "extra_runs_on_spread": "extra_runs_on_spread",
        "spread_threshold": "spread_threshold",
        "idle_seconds": "idle_seconds",
        "nvml_sample_interval_ms": "nvml_sample_interval_ms",
    }
    if include_warmup_frames:
        expected_keys["warmup_frames"] = "warmup_frames"

    errors = []
    for parameter_key, benchmark_key in expected_keys.items():
        actual = parameters.get(parameter_key)
        expected = benchmark.get(benchmark_key)
        if actual != expected:
            errors.append(
                f"{parameter_key} must match canonical {benchmark_key} "
                f"({actual!r} != {expected!r})"
            )
    return errors


def suite_publishability_errors(
    *,
    status: str,
    canonical_errors: list[str],
    runs: list[dict[str, Any]],
) -> list[str]:
    """Collect suite-level reasons that prevent publishing a result."""
    errors = list(canonical_errors)
    if status != "valid":
        errors.append(f"Suite status is {status!r}, not 'valid'")
    for run in runs:
        run_index = run.get("run_index", "?")
        for error in run.get("reproducibility", {}).get("errors", []):
            errors.append(f"Run {run_index} reproducibility: {error}")
    return list(dict.fromkeys(errors))


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
    environment: dict[str, Any],
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
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_index": run_index,
        "product": "ai-media-enhancer",
        "backend": config.backend,
        "workload_id": workload_id,
        "variant": config.variant,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "parameters": {
            "frames": config.frames,
            "warmup_frames": config.warmup_frames,
            "gpu_id": config.gpu_id,
            "bitrate_mbps": config.bitrate_mbps,
            "crf": config.crf if config.backend == "ffmpeg" else None,
            "cuda_graph": config.cuda_graph,
            "nvml_sample_interval_ms": config.sample_interval_ms,
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

    sampler.start(time.perf_counter())
    start_time = time.perf_counter()
    try:
        measured_returncode = run_child(measured_command, measured_stdout, measured_stderr)
    finally:
        end_time = time.perf_counter()
        samples = sampler.samples_relative_to(sampler.stop(), start_time)
    wall_time_sec = end_time - start_time
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
                enforce_bitrate=True,
            ),
        )
        if measured_output.is_file()
        else {"valid": False, "errors": ["Measured output was not created"]}
    )
    output_asset = _hash_if_present(measured_output, "output", root)
    metrics = {
        "wall_time_sec": wall_time_sec,
        "end_to_end_fps": config.frames / wall_time_sec if wall_time_sec > 0 else None,
        "processed_frames": config.frames,
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


def run_suite(config: BenchmarkConfig, root: Path | None = None) -> tuple[dict[str, Any], int]:
    """Run a complete 3+2 suite and return its machine-readable summary."""
    validate_config(config)
    root = (root or Path.cwd()).resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar, sidecar_path = load_engine_contract(config.engine)
    assets, workload_id = collect_assets(config, sidecar, sidecar_path, root)
    sampler = NvmlSampler(config.gpu_id, config.sample_interval_ms)
    gpu = sampler.initialize()
    environment = collect_environment(gpu)
    summary_path = config.output_dir / "suite.json"
    run_manifests: list[dict[str, Any]] = []
    target_runs = config.initial_runs

    try:
        run_index = 1
        while run_index <= target_runs:
            if run_index > 1 and config.idle_seconds > 0:
                time.sleep(config.idle_seconds)
            print(
                f"Benchmark run {run_index}/{target_runs}: {config.backend}, "
                f"{config.frames} frames",
                file=sys.stderr,
            )
            result = run_one(
                config,
                run_index=run_index,
                sidecar=sidecar,
                assets=assets,
                workload_id=workload_id,
                environment=environment,
                sampler=sampler,
                root=root,
            )
            run_manifests.append(result)
            if result.get("status") != "valid":
                report_invalid_run(result)
                break
            if run_index == config.initial_runs and config.extra_runs > 0:
                fps_values = [
                    run["measured"]["metrics"]["end_to_end_fps"] for run in run_manifests
                ]
                stats = compute_suite_statistics(fps_values)
                spread = stats["relative_spread"]
                if should_extend_suite(fps_values, config.spread_threshold):
                    target_runs += config.extra_runs
                    print(
                        f"Relative spread {spread:.2%} exceeds "
                        f"{config.spread_threshold:.2%}; extending suite to {target_runs} runs",
                        file=sys.stderr,
                    )
            run_index += 1
    finally:
        sampler.shutdown()

    valid_runs = [run for run in run_manifests if run.get("status") == "valid"]
    fps_values = [run["measured"]["metrics"]["end_to_end_fps"] for run in valid_runs]
    statistics_report = compute_suite_statistics(fps_values)
    spread = statistics_report["relative_spread"]
    suite_errors: list[str] = []
    power_limits = {
        run["measured"]["metrics"]["nvml"]["power"].get("limit_w")
        for run in valid_runs
        if run["measured"]["metrics"]["nvml"]["power"].get("limit_w") is not None
    }
    if len(power_limits) > 1:
        suite_errors.append("GPU power limit changed between measured runs")
    all_valid = len(valid_runs) == len(run_manifests) == target_runs and not suite_errors
    stable = all_valid and spread is not None and spread <= config.spread_threshold
    status = "valid" if stable else ("unstable" if all_valid else "invalid")
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
    }
    benchmark_contract = (
        load_json(config.workload_manifest).get("benchmark")
        if config.workload_manifest is not None
        else None
    )
    canonical_errors = canonical_suite_errors(
        parameters,
        benchmark_contract if isinstance(benchmark_contract, dict) else None,
        include_warmup_frames=True,
    )
    publishability_errors = suite_publishability_errors(
        status=status,
        canonical_errors=canonical_errors,
        runs=run_manifests,
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "publishable": not publishability_errors,
        "publishability": {
            "canonical_contract": not canonical_errors,
            "errors": publishability_errors,
        },
        "product": "ai-media-enhancer",
        "backend": config.backend,
        "workload_id": workload_id,
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
    if publishability_errors:
        print("Benchmark suite is not publishable:", file=sys.stderr)
        for error in publishability_errors:
            print(f"  - {error}", file=sys.stderr)
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
