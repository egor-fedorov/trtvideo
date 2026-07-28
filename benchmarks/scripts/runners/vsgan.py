#!/usr/bin/env python3
"""Plan or run a pinned VSGAN video benchmark profile."""

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
from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
    validate_vsgan_engine_contract,
)
from benchmarks.scripts.runners.vapoursynth_profile import (
    VapourSynthExecutionProfile,
    add_execution_profile_arguments,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runners.vapoursynth_suite import (
    VapourSynthImplementation,
    VapourSynthSuiteConfig,
    VapourSynthWorkload,
    run_vapoursynth_suite,
)
from benchmarks.scripts.runners.vspipe_nvenc import VspipeNvencConfig
from benchmarks.scripts.runtime.command import CommandSpec, display_command
from benchmarks.scripts.runtime.io import write_json_target, write_summary_target
from benchmarks.scripts.runtime.suite import SuitePolicy
from benchmarks.scripts.workloads.manifest import (
    WorkloadError,
    find_clip_variant,
    find_model_variant,
)
from trtvideo.video.nvcodec.encoder import NvencCbrContract

VSGAN_SCRIPT = "/app/benchmarks/vsgan/upscale.vpy"


def build_vsgan_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
    frames: int,
    source: str | None = None,
    profile: VapourSynthExecutionProfile | None = None,
) -> CommandSpec:
    """Build the pinned VSGAN vspipe-to-NVENC command pipeline."""
    profile = profile or resolve_execution_profile(args, "vsgan")
    variant = find_clip_variant(manifest, args.variant)
    bitrate_mbps = variant["benchmark_output"]["bitrate_mbps"]
    fps_num, fps_den = (int(part) for part in manifest["clip"]["fps"].split("/", 1))
    gop = max(1, round(fps_num / fps_den))
    encoder = NvencCbrContract(
        bitrate_bps=int(bitrate_mbps * 1_000_000),
        gop_frames=gop,
    )
    return VspipeNvencConfig(
        script=VSGAN_SCRIPT,
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
    """Build a machine-readable pinned VSGAN plan."""
    profile = profile or resolve_execution_profile(args, "vsgan")
    manifest = load_json(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, "vsgan")
    validate_declared_profile(implementation, profile)
    measured_implementation = dict(implementation)
    result_class = profile.mode
    measured_implementation["comparison_class"] = result_class
    measured_implementation["role"] = "product"
    parameters = benchmark_parameters(args, manifest)
    variant = find_clip_variant(manifest, args.variant)
    input_path = str(Path("/app") / variant["path"])
    output_dir = Path(args.output_dir)
    warmup = build_vsgan_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-warmup.mp4",
        frames=parameters["warmup_frames"],
        source=input_path,
        profile=profile,
    )
    measured = build_vsgan_command(
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
        product="VSGAN-tensorrt-docker",
        backend="VapourSynth/vstrt",
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
            "Pinned VSGAN uses TensorRT 10.16 and therefore receives a separate engine.",
            "The engine must be built from the same canonical ONNX on the same GPU.",
            "The mounted .vpy file is configuration only; no VSGAN source is patched.",
        ],
    )
    return plan, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark pinned VSGAN")
    add_common_arguments(parser, engine=True)
    add_execution_profile_arguments(parser)
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        profile = resolve_execution_profile(args, "vsgan")
        plan, manifest = build_plan(args, profile=profile)
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        engine = Path(args.engine)
        sidecar, sidecar_path = load_engine_contract(engine)
        parameters = plan["parameters"]
        variant = find_clip_variant(manifest, args.variant)
        input_path = Path("/app") / variant["path"]
        onnx_path = Path(find_model_variant(manifest, args.variant)["fp16_path"])
        validate_vsgan_engine_contract(
            sidecar,
            manifest,
            args.variant,
            onnx_path,
            str(plan["implementation"]["upstream_image"]),
            str(plan["implementation"]["encoder_ffmpeg_package"]),
        )
        lock_path = Path(manifest["lock_path"])
        build_log = Path(str(sidecar.get("build_log", "")))
        if not build_log.is_file():
            raise CompetitorError(f"VSGAN engine build log not found: {build_log}")
        config = VapourSynthSuiteConfig(
            implementation=VapourSynthImplementation(
                product="VSGAN-tensorrt-docker",
                backend="VapourSynth/vstrt",
                comparison_class=plan["comparison_class"],
                metadata=plan["implementation"],
                max_compute_processes=parameters["max_compute_processes"],
                max_graphics_processes=parameters["max_graphics_processes"],
            ),
            workload=VapourSynthWorkload(
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
                    "engine_build_log": build_log,
                    "asset_lock": lock_path,
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
            warmup_command=lambda path, frames: build_vsgan_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
                source=str(input_path),
                profile=profile,
            ),
            measured_command=lambda path, frames: build_vsgan_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
                source=str(input_path),
                profile=profile,
            ),
            keep_outputs=args.keep_outputs,
        )
        summary, returncode = run_vapoursynth_suite(config)
        write_summary_target(args.json, summary)
    except (
        CompetitorError,
        EngineContractError,
        OSError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
