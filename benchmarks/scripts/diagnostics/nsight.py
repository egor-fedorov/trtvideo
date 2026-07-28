#!/usr/bin/env python3
"""Capture one reproducible Nsight Systems trace of the project pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.benchmark import (
    asset_requirement,
    load_json,
    plan_document,
)
from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
)
from benchmarks.scripts.runners.trtvideo_suite import (
    BenchmarkConfig,
    BenchmarkError,
    collect_assets,
    output_contract,
    validate_config,
)
from benchmarks.scripts.runtime.environment import (
    collect_environment,
    environment_errors,
    relative_artifact_path,
    sanitize_command,
    write_json,
)
from benchmarks.scripts.runtime.io import write_json_target
from benchmarks.scripts.runtime.nvml import NvmlError, NvmlSampler
from benchmarks.scripts.workloads.manifest import WorkloadError, find_clip_variant
from trtvideo.benchmarking.validation import validate_output
from trtvideo.diagnostics.nvtx import NVTX_ENV

TRACE_APIS = "cuda,nvtx,osrt,nvvideo"
STATS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "nvvideo_api_sum",
    "osrt_sum",
    "nvtx_pushpop_sum",
    "nvtx_gpu_proj_sum",
)
OPTIONAL_EMPTY_STATS_REPORTS = {
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
}


class NsightError(RuntimeError):
    """Raised when a diagnostic trace cannot satisfy its contract."""


@dataclass(frozen=True)
class NsightPaths:
    """Canonical output paths for one diagnostic capture."""

    output_dir: Path
    trace_base: Path
    trace: Path
    sqlite: Path
    video: Path
    manifest: Path
    stdout: Path
    stderr: Path
    status: Path
    video_devices: Path
    stats_dir: Path
    stats_stderr: Path

    @classmethod
    def create(cls, output_dir: Path) -> NsightPaths:
        return cls(
            output_dir=output_dir,
            trace_base=output_dir / "trtvideo",
            trace=output_dir / "trtvideo.nsys-rep",
            sqlite=output_dir / "trtvideo.sqlite",
            video=output_dir / "output.mp4",
            manifest=output_dir / "manifest.json",
            stdout=output_dir / "nsys.stdout.log",
            stderr=output_dir / "nsys.stderr.log",
            status=output_dir / "nsys-status.txt",
            video_devices=output_dir / "gpu-video-devices.txt",
            stats_dir=output_dir / "stats",
            stats_stderr=output_dir / "stats.stderr.log",
        )


def build_upscale_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
) -> list[str]:
    """Build the ordinary unprofiled pipeline command wrapped by Nsight."""
    variant = find_clip_variant(manifest, args.variant)
    bitrate_mbps = variant["benchmark_output"]["bitrate_mbps"]
    return [
        "upscale",
        "--engine",
        args.engine,
        "--input",
        variant["path"],
        "--output",
        str(output_path),
        "--gpu-id",
        str(args.gpu_id),
        "--max-frames",
        str(args.frames),
        "--codec",
        "h264",
        "--bitrate-mbps",
        str(bitrate_mbps),
        "--quiet",
    ]


def build_nsight_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    paths: NsightPaths,
) -> list[str]:
    """Build the exact Nsight CLI invocation for the diagnostic trace."""
    return [
        args.nsys,
        "profile",
        f"--trace={TRACE_APIS}",
        "--sample=none",
        "--cpuctxsw=none",
        f"--gpu-video-devices={args.gpu_id}",
        "--force-overwrite=true",
        "--output",
        str(paths.trace_base),
        *build_upscale_command(args, manifest, output_path=paths.video),
    ]


def build_stats_command(
    nsys: str,
    report: str,
    sqlite: Path,
) -> list[str]:
    """Build one stable CSV summary command for a captured report."""
    return [
        nsys,
        "stats",
        "--quiet",
        "--report",
        report,
        "--format",
        "csv",
        str(sqlite),
    ]


def build_export_command(nsys: str, paths: NsightPaths) -> list[str]:
    """Build one deterministic SQLite export for all stats reports."""
    return [
        nsys,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--quiet=true",
        "--output",
        str(paths.sqlite),
        str(paths.trace),
    ]


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable diagnostic plan without requiring GPU assets."""
    if args.frames <= 0:
        raise NsightError("--frames must be greater than zero")
    manifest = load_json(Path(args.manifest))
    paths = NsightPaths.create(Path(args.output_dir))
    command = build_nsight_command(args, manifest, paths=paths)
    plan = plan_document(
        product="trtvideo",
        backend="nvcodec/Nsight Systems",
        implementation={
            "role": "diagnostic",
            "image": os.environ.get("TRTVIDEO_IMAGE_REF", "trtvideo:benchmark"),
            "profiler": "Nsight Systems",
        },
        manifest=manifest,
        variant_name=args.variant,
        parameters={
            "frames": args.frames,
            "gpu_id": args.gpu_id,
            "trace_apis": TRACE_APIS.split(","),
            "gpu_video_trace": True,
            "cpu_sampling": False,
            "cpu_context_switch_trace": False,
            "cuda_graph": False,
            "nvtx_environment": {NVTX_ENV: "1"},
            "stats_reports": list(STATS_REPORTS),
        },
        commands={"profile": [command]},
        assets=[
            asset_requirement(args.engine, "engine"),
            asset_requirement(args.manifest, "workload_manifest"),
        ],
        limitations=[
            "Profiler overhead makes trace FPS non-publishable.",
            "CPU IP sampling and scheduler context-switch tracing are disabled.",
            "This diagnostic covers only the trtvideo nvcodec pipeline.",
        ],
    )
    return plan, manifest


def _run_text(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def gpu_video_preflight_error(
    returncode: int,
    output: str,
    *,
    gpu_id: int,
) -> str | None:
    """Return a useful error when Nsight cannot trace GPU video accelerators."""
    normalized = output.lower()
    if returncode != 0:
        return (
            "Nsight GPU video tracing check failed with code "
            f"{returncode}; see gpu-video-devices.txt"
        )
    if (
        "could not find any nvidia gpus" in normalized
        or "gpu video accelerator tracing is not available" in normalized
    ):
        return (
            f"Nsight cannot trace GPU video accelerators on GPU {gpu_id}; "
            "see gpu-video-devices.txt"
        )
    return None


def _write_preflight(
    paths: NsightPaths,
    nsys: str,
    *,
    gpu_id: int,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    version_code, version_stdout, version_stderr = _run_text([nsys, "--version"])
    version = (version_stdout.strip() or version_stderr.strip()).splitlines()
    version_text = version[0] if version else "unknown"
    if version_code != 0:
        errors.append(f"Nsight version check failed with code {version_code}")

    status_code, status_stdout, status_stderr = _run_text(
        [nsys, "status", "--environment"]
    )
    paths.status.write_text(status_stdout + status_stderr, encoding="utf-8")
    if status_code != 0:
        errors.append(f"Nsight environment check failed with code {status_code}")

    devices_code, devices_stdout, devices_stderr = _run_text(
        [nsys, "profile", "--gpu-video-devices=help"]
    )
    paths.video_devices.write_text(
        devices_stdout + devices_stderr,
        encoding="utf-8",
    )
    devices_error = gpu_video_preflight_error(
        devices_code,
        devices_stdout + devices_stderr,
        gpu_id=gpu_id,
    )
    if devices_error is not None:
        errors.append(devices_error)
    return version_text, errors


def _run_profile(
    command: list[str],
    *,
    paths: NsightPaths,
) -> int:
    environment = dict(os.environ)
    environment[NVTX_ENV] = "1"
    try:
        with (
            paths.stdout.open("w", encoding="utf-8") as stdout,
            paths.stderr.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
            )
    except OSError as exc:
        paths.stderr.write_text(f"Failed to start Nsight: {exc}\n", encoding="utf-8")
        return 127
    return result.returncode


def _write_stats(paths: NsightPaths, nsys: str) -> tuple[dict[str, str], list[str]]:
    artifacts: dict[str, str] = {}
    errors: list[str] = []
    stderr_chunks: list[str] = []
    paths.stats_dir.mkdir(parents=True, exist_ok=True)
    export_code, export_stdout, export_stderr = _run_text(
        build_export_command(nsys, paths)
    )
    if export_stdout or export_stderr:
        stderr_chunks.append(
            f"== sqlite export ==\n{export_stdout}{export_stderr}".rstrip() + "\n"
        )
    if export_code != 0:
        errors.append(f"Nsight SQLite export failed with code {export_code}")
    elif not paths.sqlite.is_file():
        errors.append("Nsight SQLite export was not created")

    for report in STATS_REPORTS if not errors else ():
        command = build_stats_command(nsys, report, paths.sqlite)
        returncode, stdout, stderr = _run_text(command)
        report_path = paths.stats_dir / f"{report}.csv"
        report_path.write_text(stdout, encoding="utf-8")
        artifacts[report] = str(report_path)
        if stderr:
            stderr_chunks.append(f"== {report} ==\n{stderr.rstrip()}\n")
        if returncode != 0:
            errors.append(f"Nsight stats report {report} failed with code {returncode}")
        elif (
            report not in OPTIONAL_EMPTY_STATS_REPORTS
            and (not stdout.strip() or "SKIPPED:" in stdout or "SKIPPED:" in stderr)
        ):
            errors.append(f"Nsight stats report {report} contains no data")
    paths.stats_stderr.write_text("\n".join(stderr_chunks), encoding="utf-8")
    return artifacts, errors


def _diagnostic_config(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    paths: NsightPaths,
) -> BenchmarkConfig:
    variant = find_clip_variant(manifest, args.variant)
    return BenchmarkConfig(
        engine=Path(args.engine),
        input_path=Path(variant["path"]),
        output_dir=paths.output_dir,
        gpu_id=args.gpu_id,
        frames=args.frames,
        warmup_frames=1,
        initial_runs=1,
        extra_runs=0,
        idle_seconds=0,
        bitrate_mbps=float(variant["benchmark_output"]["bitrate_mbps"]),
        workload_manifest=Path(args.manifest),
        variant=args.variant,
        keep_outputs=True,
    )


def run_diagnostic(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Capture, summarize, and validate one Nsight diagnostic."""
    started_at_utc = datetime.now(UTC).isoformat()
    root = (root or Path.cwd()).resolve()
    paths = NsightPaths.create(Path(args.output_dir))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    config = _diagnostic_config(args, manifest, paths)
    validate_config(config)
    sidecar, sidecar_path = load_engine_contract(config.engine)
    assets, workload_id = collect_assets(config, sidecar, sidecar_path, root)

    sampler = NvmlSampler(args.gpu_id, 100)
    try:
        gpu = sampler.initialize()
        environment = collect_environment(gpu)
    finally:
        sampler.shutdown()

    nsys_version, errors = _write_preflight(
        paths,
        args.nsys,
        gpu_id=args.gpu_id,
    )
    environment["software"]["nsight_systems"] = nsys_version
    errors.extend(environment_errors(environment))
    profile_command = build_nsight_command(args, manifest, paths=paths)
    profile_returncode = 1
    stats_artifacts: dict[str, str] = {}
    validation: dict[str, Any] = {
        "valid": False,
        "errors": ["Profile did not run"],
    }

    if not errors:
        profile_returncode = _run_profile(profile_command, paths=paths)
        if profile_returncode != 0:
            errors.append(f"Nsight profile exited with code {profile_returncode}")
        if not paths.trace.is_file():
            errors.append("Nsight report was not created")
        if paths.trace.is_file():
            stats_artifacts, stats_errors = _write_stats(paths, args.nsys)
            errors.extend(stats_errors)
        if paths.video.is_file():
            validation = validate_output(
                paths.video,
                output_contract(
                    config,
                    sidecar,
                    frames=args.frames,
                    enforce_bitrate=False,
                ),
            )
        else:
            validation = {
                "valid": False,
                "errors": ["Profiled output was not created"],
            }
        if not validation.get("valid"):
            errors.extend(str(error) for error in validation.get("errors", []))

    manifest_path = paths.manifest
    result = {
        "schema_version": 1,
        "document_type": "nsight-diagnostic",
        "scope": "diagnostic",
        "publishable": False,
        "status": "valid" if not errors else "invalid",
        "product": "trtvideo",
        "workload_id": workload_id,
        "benchmark_contract_version": manifest["benchmark"]["contract_version"],
        "variant": args.variant,
        "started_at_utc": started_at_utc,
        "parameters": {
            "frames": args.frames,
            "gpu_id": args.gpu_id,
            "trace_apis": TRACE_APIS.split(","),
            "gpu_video_trace": True,
            "cpu_sampling": False,
            "cpu_context_switch_trace": False,
            "cuda_graph": False,
            "nvtx": True,
        },
        "commands": {
            "profile": sanitize_command(profile_command, root),
            "environment": {NVTX_ENV: "1"},
        },
        "assets": assets,
        "environment": environment,
        "profile": {
            "returncode": profile_returncode,
            "output_validation": validation,
        },
        "artifacts": {
            "manifest": relative_artifact_path(manifest_path, root),
            "trace": (
                relative_artifact_path(paths.trace, root)
                if paths.trace.is_file()
                else None
            ),
            "sqlite": (
                relative_artifact_path(paths.sqlite, root)
                if paths.sqlite.is_file()
                else None
            ),
            "output": (
                relative_artifact_path(paths.video, root)
                if paths.video.is_file()
                else None
            ),
            "stdout": (
                relative_artifact_path(paths.stdout, root)
                if paths.stdout.is_file()
                else None
            ),
            "stderr": (
                relative_artifact_path(paths.stderr, root)
                if paths.stderr.is_file()
                else None
            ),
            "environment_status": relative_artifact_path(paths.status, root),
            "gpu_video_devices": relative_artifact_path(paths.video_devices, root),
            "stats_stderr": (
                relative_artifact_path(paths.stats_stderr, root)
                if paths.stats_stderr.is_file()
                else None
            ),
            "stats": {
                name: relative_artifact_path(Path(path), root)
                for name, path in stats_artifacts.items()
            },
        },
        "limitations": [
            "Profiler overhead makes trace FPS non-publishable.",
            "CPU IP sampling and scheduler context-switch tracing are disabled.",
        ],
        "errors": list(dict.fromkeys(errors)),
    }
    write_json(manifest_path, result)
    return result, 0 if result["status"] == "valid" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one Nsight Systems trace of trtvideo",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        plan, manifest = build_plan(args)
        if args.dry_run:
            write_json_target(plan, args.json)
            return
        result, returncode = run_diagnostic(args, manifest)
        if args.json is not None:
            write_json_target(result, args.json)
        print(
            f"Nsight diagnostic {result['status']}: {args.output_dir}",
            file=sys.stderr,
        )
    except (
        BenchmarkError,
        EngineContractError,
        NsightError,
        NvmlError,
        OSError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
