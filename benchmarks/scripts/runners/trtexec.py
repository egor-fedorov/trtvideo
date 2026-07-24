#!/usr/bin/env python3
"""Plan or run the TensorRT 11 trtexec inference-ceiling benchmark."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_media.benchmarking.cpu import snapshot_child_cpu, summarize_child_cpu
from ai_media.benchmarking.environment import (
    collect_environment,
    environment_errors,
    relative_artifact_path,
    sha256_file,
    write_json,
)
from ai_media.benchmarking.nvml import NvmlSampler, summarize_samples, write_samples
from ai_media.benchmarking.runner import (
    load_engine_contract,
    write_summary_target,
)
from ai_media.benchmarking.suite import (
    SuitePolicy,
    SuiteRunner,
    canonical_suite_errors,
    report_publishability_errors,
    suite_publishability_errors,
)
from benchmarks.scripts.runners.common import (
    CompetitorError,
    asset_requirement,
    benchmark_parameters,
    find_variant,
    implementation_config,
    load_json,
    plan_document,
    write_json_target,
)

NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def build_trtexec_command(
    args: argparse.Namespace,
    *,
    export_times: Path,
    iterations: int,
) -> list[str]:
    """Build an exact-iteration TensorRT 11 trtexec command."""
    command = [
        "trtexec",
        f"--loadEngine={args.engine}",
        f"--device={args.gpu_id}",
        f"--warmUp={args.warmup_ms}",
        "--duration=0",
        f"--iterations={iterations}",
        "--avgRuns=1",
        "--percentile=50,95,99",
        f"--exportTimes={export_times}",
    ]
    if not args.cuda_graph:
        command.append("--noCudaGraph")
    return command


def parse_trtexec_output(output: str) -> dict[str, float]:
    """Parse stable TensorRT summary fields without depending on table layout."""
    patterns = {
        "throughput_qps": rf"Throughput:\s*{NUMBER}\s*qps",
        "latency_median_ms": rf"Latency:.*?median\s*=\s*{NUMBER}\s*ms",
        "latency_p95_ms": rf"Latency:.*?percentile\(95%\)\s*=\s*{NUMBER}\s*ms",
        "gpu_compute_median_ms": rf"GPU Compute Time:.*?median\s*=\s*{NUMBER}\s*ms",
        "gpu_compute_p95_ms": (
            rf"GPU Compute Time:.*?percentile\(95%\)\s*=\s*{NUMBER}\s*ms"
        ),
    }
    metrics: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.DOTALL)
        if match is None:
            raise CompetitorError(f"trtexec output has no {name}")
        metrics[name] = float(match.group(1))
    return metrics


def _implementation_environment(
    implementation: dict[str, Any],
    gpu: dict[str, Any],
) -> dict[str, Any]:
    environment = collect_environment(gpu)
    environment["image"] = {
        "reference": os.environ.get("AI_MEDIA_IMAGE_REF", implementation["image"]),
        "id": os.environ.get("AI_MEDIA_IMAGE_ID", "unknown"),
        "base_reference": os.environ.get("AI_MEDIA_BASE_IMAGE", implementation["source"]),
        "repository_revision": os.environ.get("AI_MEDIA_BUILD_REVISION", "unknown"),
        "source_dirty": os.environ.get("AI_MEDIA_BUILD_DIRTY", "unknown"),
    }
    environment["implementation"] = implementation
    return environment


def _run_trtexec(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    start = time.perf_counter()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    except OSError as exc:
        stderr_path.write_text(f"Failed to start trtexec: {exc}\n", encoding="utf-8")
        return 127, time.perf_counter() - start
    return result.returncode, time.perf_counter() - start


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable trtexec plan and return the workload manifest."""
    manifest = load_json(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, "trtexec")
    args.warmup_frames = manifest["benchmark"]["warmup_frames"]
    parameters = benchmark_parameters(args, manifest)
    parameters.pop("warmup_frames")
    parameters.update(
        {
            "warmup_ms": args.warmup_ms,
            "cuda_graph": args.cuda_graph,
            "data_transfers": False,
        }
    )
    command = build_trtexec_command(
        args,
        export_times=Path(args.output_dir) / "run-NN" / "trtexec.times.json",
        iterations=parameters["frames"],
    )
    plan = plan_document(
        product="trtexec",
        backend="TensorRT",
        comparison_class=implementation["comparison_class"],
        implementation=implementation,
        manifest=manifest,
        variant_name=args.variant,
        parameters=parameters,
        commands={"measured": [command]},
        assets=[
            asset_requirement(args.engine, "engine"),
            asset_requirement(args.manifest, "workload_manifest"),
        ],
        limitations=[
            "trtexec excludes video decode, colorspace conversion, encode and mux.",
            "NVML process samples include engine setup and millisecond-based warmup; "
            "throughput comes from trtexec's measured iterations.",
        ],
    )
    return plan, manifest


@dataclass(frozen=True)
class TrtexecRunContext:
    """Immutable dependencies required to execute one diagnostic run."""

    args: argparse.Namespace
    plan: dict[str, Any]
    workload: dict[str, Any]
    parameters: dict[str, Any]
    output_dir: Path
    sampler: NvmlSampler
    environment: dict[str, Any]
    assets: dict[str, Any]
    root: Path


def _run_one(context: TrtexecRunContext, run_index: int) -> dict[str, Any]:
    parameters = context.parameters
    run_dir = context.output_dir / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "trtexec.stdout.log"
    stderr_path = run_dir / "trtexec.stderr.log"
    times_path = run_dir / "trtexec.times.json"
    samples_path = run_dir / "nvml.samples.jsonl"
    manifest_path = run_dir / "manifest.json"
    command = build_trtexec_command(
        context.args,
        export_times=times_path,
        iterations=parameters["frames"],
    )
    cpu_before = snapshot_child_cpu()
    context.sampler.start(time.perf_counter())
    sample_origin = time.perf_counter()
    try:
        returncode, wall_time = _run_trtexec(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    finally:
        cpu_after = snapshot_child_cpu()
        samples = context.sampler.samples_relative_to(
            context.sampler.stop(), sample_origin
        )
    cpu_summary = summarize_child_cpu(
        cpu_before,
        cpu_after,
        wall_time_sec=wall_time,
    ).as_dict()
    write_samples(samples_path, samples)
    combined_output = "\n".join(
        (
            stdout_path.read_text(encoding="utf-8", errors="replace"),
            stderr_path.read_text(encoding="utf-8", errors="replace"),
        )
    )
    errors = []
    parsed: dict[str, float] = {}
    if returncode != 0:
        errors.append(f"trtexec exited with code {returncode}")
    else:
        try:
            parsed = parse_trtexec_output(combined_output)
        except CompetitorError as exc:
            errors.append(str(exc))
    nvml = summarize_samples(
        samples,
        wall_time_sec=wall_time,
        frames=parameters["frames"],
        max_compute_processes=1,
        max_graphics_processes=0,
    )
    if not nvml.get("valid"):
        errors.extend(nvml.get("errors", []))
    reproducibility_errors = environment_errors(context.environment)
    errors.extend(reproducibility_errors)
    run_manifest = {
        "schema_version": 1,
        "status": "valid" if not errors else "invalid",
        "run_index": run_index,
        "product": "trtexec",
        "backend": "TensorRT",
        "comparison_class": context.plan["comparison_class"],
        "workload_id": context.workload["id"],
        "variant": context.args.variant,
        "implementation": context.plan["implementation"],
        "parameters": parameters,
        "command": command,
        "assets": context.assets,
        "environment": context.environment,
        "metrics": {
            **parsed,
            "process_wall_time_sec": wall_time,
            "processed_iterations": parameters["frames"],
            "cpu": cpu_summary,
            "nvml": nvml,
            "nvml_scope": "process-including-setup-and-warmup",
        },
        "reproducibility": {
            "publishable": not reproducibility_errors,
            "errors": reproducibility_errors,
        },
        "artifacts": {
            "manifest": relative_artifact_path(manifest_path, context.root),
            "stdout": relative_artifact_path(stdout_path, context.root),
            "stderr": relative_artifact_path(stderr_path, context.root),
            "times": relative_artifact_path(times_path, context.root),
            "nvml_samples": relative_artifact_path(samples_path, context.root),
        },
        "errors": errors,
    }
    write_json(manifest_path, run_manifest)
    return run_manifest


def _throughput_qps(manifest: dict[str, Any]) -> float:
    value = manifest.get("metrics", {}).get("throughput_qps")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompetitorError("Valid trtexec run has no throughput_qps metric")
    return float(value)


def _power_limit(manifest: dict[str, Any]) -> float | None:
    value = (
        manifest.get("metrics", {})
        .get("nvml", {})
        .get("power", {})
        .get("limit_w")
    )
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompetitorError("trtexec run has an invalid GPU power limit")
    return float(value)


def _run_suite(
    args: argparse.Namespace,
    plan: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    parameters = plan["parameters"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Path(args.engine)
    sidecar, sidecar_path = load_engine_contract(engine)
    variant = find_variant(manifest, args.variant)
    expected_output = variant["benchmark_output"]
    if sidecar["output"]["shape"][2:] != [
        expected_output["height"],
        expected_output["width"],
    ]:
        raise CompetitorError("Engine output shape does not match workload variant")

    sampler = NvmlSampler(args.gpu_id, parameters["nvml_sample_interval_ms"])
    gpu = sampler.initialize()
    environment = _implementation_environment(plan["implementation"], gpu)
    assets = {
        "engine": {
            "path": relative_artifact_path(engine, Path.cwd()),
            "sha256": sha256_file(engine),
            "size_bytes": engine.stat().st_size,
        },
        "engine_manifest": {
            "path": relative_artifact_path(sidecar_path, Path.cwd()),
            "sha256": sha256_file(sidecar_path),
            "size_bytes": sidecar_path.stat().st_size,
        },
    }
    root = Path.cwd()
    context = TrtexecRunContext(
        args=args,
        plan=plan,
        workload=manifest,
        parameters=parameters,
        output_dir=output_dir,
        sampler=sampler,
        environment=environment,
        assets=assets,
        root=root,
    )
    suite_runner = SuiteRunner(
        SuitePolicy.from_parameters(parameters),
        label="trtexec",
        frames=parameters["frames"],
        metric_reader=_throughput_qps,
        power_limit_reader=_power_limit,
    )
    try:
        suite_result = suite_runner.execute(lambda index: _run_one(context, index))
    finally:
        sampler.shutdown()

    runs = list(suite_result.runs)
    statistics = suite_result.statistics
    spread = statistics["relative_spread"]
    status = suite_result.status
    canonical_errors = canonical_suite_errors(
        parameters,
        manifest["benchmark"],
        include_warmup_frames=False,
    )
    publishability_errors = suite_publishability_errors(
        status=status,
        canonical_errors=canonical_errors,
        runs=runs,
    )
    summary = {
        "schema_version": 1,
        "document_type": "benchmark-result",
        "status": status,
        "publishable": not publishability_errors,
        "publishability": {
            "canonical_contract": not canonical_errors,
            "errors": publishability_errors,
        },
        "product": "trtexec",
        "backend": "TensorRT",
        "comparison_class": plan["comparison_class"],
        "workload_id": manifest["id"],
        "variant": args.variant,
        "implementation": plan["implementation"],
        "parameters": parameters,
        "statistics": statistics,
        "errors": list(suite_result.errors),
        "runs": [
            {
                "index": run["run_index"],
                "status": run["status"],
                "manifest": run["artifacts"]["manifest"],
                "throughput_qps": run["metrics"].get("throughput_qps"),
            }
            for run in runs
        ],
    }
    write_json(output_dir / "suite.json", summary)
    print(
        f"Benchmark suite {status}: median={statistics['median_fps']!r} qps, "
        f"spread={spread!r}",
        file=sys.stderr,
    )
    report_publishability_errors(
        publishability_errors,
        acceptance_only=False,
    )
    return summary, 0 if status == "valid" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT engine with trtexec")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--implementations",
        default="/app/benchmarks/implementations.json",
    )
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None, help="Plan/summary JSON path or '-'")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="Measured iterations")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--extra-runs", type=int, default=None)
    parser.add_argument("--idle-seconds", type=float, default=None)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.warmup_ms < 0:
            raise CompetitorError("--warmup-ms cannot be negative")
        plan, manifest = build_plan(args)
        if args.dry_run:
            write_json_target(plan, args.json)
            return
        summary, returncode = _run_suite(args, plan, manifest)
        write_summary_target(args.json, summary)
    except (CompetitorError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
