#!/usr/bin/env python3
"""Plan or run the canonical vs-mlrt/vstrt video benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ai_media.benchmarking.runner import load_engine_contract, write_summary_target
from benchmarks.scripts.competitor_common import (
    CommandSpec,
    CompetitorError,
    add_common_arguments,
    asset_requirement,
    benchmark_parameters,
    command_spec,
    competitor_config,
    display_command,
    find_variant,
    load_json,
    output_contract,
    plan_document,
    write_json_target,
)
from benchmarks.scripts.external_video_suite import (
    ExternalVideoSuiteConfig,
    run_external_video_suite,
)

VSTRt_SCRIPT = "/app/benchmarks/vstrt/upscale.vpy"


def build_vstrt_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
    frames: int,
) -> CommandSpec:
    """Build the vspipe-to-NVENC full video pipeline."""
    variant = find_variant(manifest, args.variant)
    bitrate_mbps = variant["benchmark_output"]["bitrate_mbps"]
    fps = manifest["clip"]["fps"]
    fps_num, fps_den = (int(part) for part in fps.split("/", 1))
    gop = max(1, round(fps_num / fps_den))
    vspipe = [
        "vspipe",
        "--container",
        "y4m",
        "--start",
        "0",
        "--end",
        str(frames - 1),
        "--requests",
        str(args.requests),
        "--arg",
        f"source={args.input}",
        "--arg",
        f"engine={args.engine}",
        "--arg",
        f"gpu_id={args.gpu_id}",
        "--arg",
        f"cuda_graph={int(args.cuda_graph)}",
        "--arg",
        f"num_streams={args.num_streams}",
        VSTRt_SCRIPT,
        "-",
    ]
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "yuv4mpegpipe",
        "-i",
        "pipe:0",
        "-frames:v",
        str(frames),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "cbr",
        "-b:v",
        f"{bitrate_mbps}M",
        "-minrate",
        f"{bitrate_mbps}M",
        "-maxrate",
        f"{bitrate_mbps}M",
        "-bufsize",
        f"{2 * bitrate_mbps}M",
        "-bf",
        "0",
        "-g",
        str(gop),
        "-forced-idr",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command_spec(vspipe, ffmpeg)


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable vstrt plan and return the workload manifest."""
    manifest = load_json(Path(args.manifest))
    competitors = load_json(Path(args.competitors))
    implementation = competitor_config(competitors, "vstrt")
    parameters = benchmark_parameters(args, manifest)
    variant = find_variant(manifest, args.variant)
    args.input = str(Path("/app") / variant["path"])
    output_dir = Path(args.output_dir)
    warmup = build_vstrt_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-warmup.mp4",
        frames=parameters["warmup_frames"],
    )
    measured = build_vstrt_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-output.mp4",
        frames=parameters["frames"],
    )
    parameters.update(
        {
            "cuda_graph": args.cuda_graph,
            "num_streams": args.num_streams,
            "vspipe_requests": args.requests,
            "bitrate_mbps": variant["benchmark_output"]["bitrate_mbps"],
        }
    )
    plan = plan_document(
        product="vs-mlrt",
        backend="vstrt",
        comparison_class=implementation["comparison_class"],
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
            asset_requirement(args.input, "input"),
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
    parser.add_argument("--requests", type=int, default=1, help="Concurrent vspipe requests")
    parser.add_argument("--num-streams", type=int, default=1, help="vstrt CUDA streams")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.requests <= 0 or args.num_streams <= 0:
            raise CompetitorError("--requests and --num-streams must be positive")
        plan, manifest = build_plan(args)
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        engine = Path(args.engine)
        sidecar, sidecar_path = load_engine_contract(engine)
        variant = find_variant(manifest, args.variant)
        output = variant["benchmark_output"]
        if sidecar["output"]["shape"][2:] != [output["height"], output["width"]]:
            raise CompetitorError("Engine output shape does not match workload variant")
        parameters = plan["parameters"]
        input_path = Path(args.input)
        config = ExternalVideoSuiteConfig(
            product="vs-mlrt",
            backend="vstrt",
            comparison_class=plan["comparison_class"],
            implementation=plan["implementation"],
            workload_id=manifest["id"],
            variant=args.variant,
            input_path=input_path,
            output_dir=Path(args.output_dir),
            frames=parameters["frames"],
            warmup_frames=parameters["warmup_frames"],
            initial_runs=parameters["initial_runs"],
            extra_runs=parameters["extra_runs_on_spread"],
            spread_threshold=parameters["spread_threshold"],
            idle_seconds=parameters["idle_seconds"],
            sample_interval_ms=parameters["nvml_sample_interval_ms"],
            gpu_id=args.gpu_id,
            output_contract=output_contract(
                manifest,
                variant,
                frames=parameters["frames"],
                enforce_bitrate=True,
            ),
            benchmark_contract=manifest["benchmark"],
            assets={
                "input": input_path,
                "engine": engine,
                "engine_manifest": sidecar_path,
                "workload_manifest": Path(args.manifest),
            },
            warmup_command=lambda path, frames: build_vstrt_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
            ),
            measured_command=lambda path, frames: build_vstrt_command(
                args,
                manifest,
                output_path=path,
                frames=frames,
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
