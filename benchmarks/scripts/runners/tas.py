#!/usr/bin/env python3
"""Plan or run a canonical TheAnimeScripter full-video benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.benchmark import (
    CompetitorError,
    add_common_arguments,
    asset_requirement,
    benchmark_parameters,
    implementation_config,
    load_json,
    output_contract,
    plan_document,
)
from benchmarks.scripts.contracts.engine import EngineContractError, load_engine_contract
from benchmarks.scripts.runners.tas_profile import (
    TasExecutionProfile,
    add_execution_profile_arguments,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runners.tas_suite import (
    PRODUCT_NAME,
    TasProcessPaths,
    TasSuiteConfig,
    run_tas_suite,
    suite_implementation_parameters,
)
from benchmarks.scripts.runtime.command import CommandSpec, command_spec, display_command
from benchmarks.scripts.runtime.io import write_json_target, write_summary_target
from benchmarks.scripts.runtime.suite import SuitePolicy
from benchmarks.scripts.workloads.manifest import (
    WorkloadError,
    find_clip_variant,
    find_model_variant,
)
from trtvideo.video.nvcodec.encoder import NvencCbrContract, gop_size_for_one_second


def build_tas_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
    frames: int,
    source: str | None = None,
    profile: TasExecutionProfile | None = None,
) -> CommandSpec:
    """Invoke the TAS adapter with a prebuilt engine and real encoded output."""
    profile = profile or resolve_execution_profile(args)
    variant = find_clip_variant(manifest, args.variant)
    model_variant = find_model_variant(manifest, args.variant)
    paths = TasProcessPaths.for_output(output_path)
    return command_spec(
        [
            sys.executable,
            "-m",
            "benchmarks.scripts.runners.tas_process",
            "--engine",
            str(args.engine),
            "--onnx",
            str(Path("/app") / model_variant["fp16_path"]),
            "--input",
            source or str(Path("/app") / variant["path"]),
            "--output",
            str(output_path),
            "--frames",
            str(frames),
            "--fps",
            manifest["clip"]["fps"],
            "--gpu-id",
            str(args.gpu_id),
            "--decode-method",
            profile.decode_method,
            "--writer",
            profile.writer,
            "--encoder-contract",
            str(paths.encoder_contract),
            "--lifecycle-output",
            str(paths.lifecycle),
            "--runtime-evidence",
            str(paths.runtime_evidence),
        ]
    )


def build_plan(
    args: argparse.Namespace,
    *,
    profile: TasExecutionProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a GPU-free dry-run plan with exact TAS I/O and encoder settings."""
    profile = profile or resolve_execution_profile(args)
    manifest = load_json(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, "tas")
    validate_declared_profile(implementation, profile)
    parameters = benchmark_parameters(args, manifest)
    variant = find_clip_variant(manifest, args.variant)
    model_variant = find_model_variant(manifest, args.variant)
    input_path = str(Path("/app") / variant["path"])
    onnx_path = str(Path("/app") / model_variant["fp16_path"])
    output_dir = Path(args.output_dir)
    warmup = build_tas_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-warmup.mp4",
        frames=parameters["warmup_frames"],
        profile=profile,
    )
    measured = build_tas_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-output.mp4",
        frames=parameters["frames"],
        profile=profile,
    )
    encoder = NvencCbrContract(
        bitrate_bps=int(variant["benchmark_output"]["bitrate_mbps"] * 1_000_000),
        gop_frames=gop_size_for_one_second(manifest["clip"]["fps"]),
    )
    parameters.update(
        {
            **profile.as_parameters(),
            "batch_size": 1,
            "full_frame": True,
            "tiling": False,
            "bitrate_mbps": variant["benchmark_output"]["bitrate_mbps"],
            "bitrate_validation": not args.skip_bitrate_validation,
            "encoder": encoder.as_dict(),
            "max_compute_processes": profile.max_compute_processes,
            "max_graphics_processes": 0,
        }
    )
    plan = plan_document(
        product=PRODUCT_NAME,
        backend="tas",
        implementation=implementation,
        manifest=manifest,
        variant_name=args.variant,
        parameters=parameters,
        commands={
            "warmup": warmup,
            "measured": measured,
            "warmup_display": display_command(warmup),
            "measured_display": display_command(measured),
        },
        assets=[
            asset_requirement(input_path, "input"),
            asset_requirement(onnx_path, "onnx"),
            asset_requirement(args.engine, "engine"),
            asset_requirement(args.manifest, "workload_manifest"),
        ],
        limitations=[
            "TAS uses the canonical ONNX with a separately prebuilt static FP32-I/O engine; "
            "engine building and model downloads are forbidden in measured processes.",
            "The encoder is normalized to the common NVENC contract for both writers; "
            "upstream-default retains CPU decoding and FFmpeg output.",
            "Actual decoder, writer, engine reuse and runtime versions must be evidenced "
            "for warmup and measurement; a silent fallback invalidates the suite.",
        ],
    )
    return plan, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TheAnimeScripter")
    add_common_arguments(parser, engine=True)
    add_execution_profile_arguments(parser)
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        profile = resolve_execution_profile(args)
        plan, manifest = build_plan(args, profile=profile)
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        from benchmarks.scripts.contracts.engine import validate_tas_engine_contract

        engine = Path(args.engine)
        sidecar, sidecar_path = load_engine_contract(engine)
        variant = find_clip_variant(manifest, args.variant)
        model_variant = find_model_variant(manifest, args.variant)
        onnx_path = Path("/app") / model_variant["fp16_path"]
        validate_tas_engine_contract(
            sidecar, manifest, args.variant, onnx_path, plan["implementation"]
        )
        parameters = plan["parameters"]
        config = TasSuiteConfig(
            implementation=plan["implementation"],
            workload_id=manifest["id"],
            variant=args.variant,
            output_dir=Path(args.output_dir),
            frames=parameters["frames"],
            warmup_frames=parameters["warmup_frames"],
            output_contract=output_contract(
                manifest,
                variant,
                frames=parameters["frames"],
                enforce_bitrate=parameters["bitrate_validation"],
            ),
            benchmark_contract=manifest["benchmark"],
            assets={
                "input": Path("/app") / variant["path"],
                "onnx": onnx_path,
                "engine": engine,
                "engine_manifest": sidecar_path,
                "asset_lock": Path("/app") / manifest["lock_path"],
                "workload_manifest": Path(args.manifest),
                "benchmark_adapter": Path(__file__).with_name("tas_process.py"),
                "benchmark_runtime_adapter": Path(__file__).with_name("tas_runtime.py"),
            },
            policy=SuitePolicy.from_parameters(parameters),
            sample_interval_ms=parameters["nvml_sample_interval_ms"],
            gpu_id=args.gpu_id,
            profile=profile,
            implementation_parameters=suite_implementation_parameters(parameters),
            runtime_versions=sidecar["runtime"],
            command=lambda path, frames: build_tas_command(
                args, manifest, output_path=path, frames=frames, profile=profile
            ),
            keep_outputs=args.keep_outputs,
        )
        summary, returncode = run_tas_suite(config)
        write_summary_target(args.json, summary)
    except (CompetitorError, EngineContractError, OSError, ValueError, WorkloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
