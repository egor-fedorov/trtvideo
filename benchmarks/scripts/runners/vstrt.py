#!/usr/bin/env python3
"""Plan or run the canonical vs-mlrt/vstrt video benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.runners.common import (
    CommandSpec,
    CompetitorError,
    add_common_arguments,
    asset_requirement,
    benchmark_parameters,
    display_command,
    find_model_variant,
    find_variant,
    implementation_config,
    load_json,
    output_contract,
    plan_document,
    validate_static_engine_contract,
    write_json_target,
)
from benchmarks.scripts.runners.external_video_suite import (
    ExternalImplementation,
    ExternalVideoSuiteConfig,
    ExternalVideoWorkload,
    run_external_video_suite,
)
from benchmarks.scripts.runners.vapoursynth_profile import (
    VapourSynthExecutionProfile,
    add_execution_profile_arguments,
    comparison_class,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runners.vspipe_nvenc import VspipeNvencConfig
from benchmarks.scripts.runtime.runner import load_engine_contract, write_summary_target
from benchmarks.scripts.runtime.suite import SuitePolicy
from trtvideo.video.nvenc import NvencCbrContract

VSTRt_SCRIPT = "/app/benchmarks/vstrt/upscale.vpy"


def build_vstrt_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
    frames: int,
    source: str | None = None,
    profile: VapourSynthExecutionProfile | None = None,
) -> CommandSpec:
    """Build the vspipe-to-NVENC full video pipeline."""
    profile = profile or resolve_execution_profile(args, "vstrt")
    variant = find_variant(manifest, args.variant)
    bitrate_mbps = variant["benchmark_output"]["bitrate_mbps"]
    fps = manifest["clip"]["fps"]
    fps_num, fps_den = (int(part) for part in fps.split("/", 1))
    gop = max(1, round(fps_num / fps_den))
    encoder = NvencCbrContract(
        bitrate_bps=int(bitrate_mbps * 1_000_000),
        gop_frames=gop,
    )
    return VspipeNvencConfig(
        script=VSTRt_SCRIPT,
        source=source or args.input,
        engine=args.engine,
        gpu_id=args.gpu_id,
        requests=profile.requests,
        cuda_graph=profile.cuda_graph,
        num_streams=profile.num_streams,
        encoder=encoder,
        script_arguments=(
            (("vs_threads", str(profile.vapoursynth_threads)),)
            if profile.vapoursynth_threads is not None
            else ()
        ),
    ).build(output_path=output_path, frames=frames)


def build_plan(
    args: argparse.Namespace,
    *,
    profile: VapourSynthExecutionProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable vstrt plan and return the workload manifest."""
    profile = profile or resolve_execution_profile(args, "vstrt")
    manifest = load_json(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, "vstrt")
    validate_declared_profile(implementation, profile)
    measured_implementation = dict(implementation)
    result_class = comparison_class(
        str(implementation["comparison_class"]),
        profile,
    )
    measured_implementation["comparison_class"] = result_class
    if profile.mode != "parity":
        measured_implementation["role"] = "technical"
    parameters = benchmark_parameters(args, manifest)
    variant = find_variant(manifest, args.variant)
    input_path = str(Path("/app") / variant["path"])
    output_dir = Path(args.output_dir)
    warmup = build_vstrt_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-warmup.mp4",
        frames=parameters["warmup_frames"],
        source=input_path,
        profile=profile,
    )
    measured = build_vstrt_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-output.mp4",
        frames=parameters["frames"],
        source=input_path,
        profile=profile,
    )
    fps_num, fps_den = (int(part) for part in manifest["clip"]["fps"].split("/", 1))
    encoder = NvencCbrContract(
        bitrate_bps=int(variant["benchmark_output"]["bitrate_mbps"] * 1_000_000),
        gop_frames=max(1, round(fps_num / fps_den)),
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
            "max_compute_processes": 2,
            "max_graphics_processes": 0,
        }
    )
    plan = plan_document(
        product="vs-mlrt",
        backend="vstrt",
        comparison_class=result_class,
        implementation=measured_implementation,
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
            asset_requirement(args.engine, "engine"),
            asset_requirement(args.manifest, "workload_manifest"),
        ],
        limitations=[
            "The vstrt plugin is rebuilt against TensorRT 11; engine loading remains "
            "a GPU acceptance gate.",
            "BestSource decode and vspipe/FFmpeg frame transfer are part of end-to-end timing.",
        ],
    )
    return plan, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark vs-mlrt/vstrt")
    add_common_arguments(parser, engine=True)
    add_execution_profile_arguments(parser)
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        profile = resolve_execution_profile(args, "vstrt")
        plan, manifest = build_plan(args, profile=profile)
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        engine = Path(args.engine)
        sidecar, sidecar_path = load_engine_contract(engine)
        variant = find_variant(manifest, args.variant)
        model_variant = find_model_variant(manifest, args.variant)
        onnx_path = Path(model_variant["fp16_path"])
        validate_static_engine_contract(
            sidecar,
            manifest,
            args.variant,
            onnx_path,
        )
        parameters = plan["parameters"]
        input_path = Path("/app") / variant["path"]
        config = ExternalVideoSuiteConfig(
            implementation=ExternalImplementation(
                product="vs-mlrt",
                backend="vstrt",
                comparison_class=plan["comparison_class"],
                metadata=plan["implementation"],
                max_compute_processes=parameters["max_compute_processes"],
                max_graphics_processes=parameters["max_graphics_processes"],
            ),
            workload=ExternalVideoWorkload(
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
                    "input": input_path,
                    "onnx": onnx_path,
                    "engine": engine,
                    "engine_manifest": sidecar_path,
                    "asset_lock": Path(manifest["lock_path"]),
                    "workload_manifest": Path(args.manifest),
                },
            ),
            policy=SuitePolicy.from_parameters(parameters),
            sample_interval_ms=parameters["nvml_sample_interval_ms"],
            gpu_id=args.gpu_id,
            implementation_parameters={
                **profile.as_parameters(),
                "batch_size": 1,
                "full_frame": True,
                "tiling": False,
                "encoder": parameters["encoder"],
            },
            warmup_command=lambda path, frames: build_vstrt_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
                source=str(input_path),
                profile=profile,
            ),
            measured_command=lambda path, frames: build_vstrt_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
                source=str(input_path),
                profile=profile,
            ),
            keep_outputs=args.keep_outputs,
        )
        summary, returncode = run_external_video_suite(config)
        write_summary_target(args.json, summary)
    except (CompetitorError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
