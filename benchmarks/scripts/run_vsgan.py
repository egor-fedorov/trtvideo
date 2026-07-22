#!/usr/bin/env python3
"""Plan or run the pinned stock VSGAN full-video benchmark."""

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
from benchmarks.scripts.external_video_suite import (
    ExternalVideoSuiteConfig,
    run_external_video_suite,
)

VSGAN_SCRIPT = "/app/benchmarks/vsgan/upscale.vpy"


def build_vsgan_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    output_path: Path,
    frames: int,
) -> CommandSpec:
    """Build the stock VSGAN vspipe-to-NVENC command pipeline."""
    variant = find_variant(manifest, args.variant)
    bitrate_mbps = variant["benchmark_output"]["bitrate_mbps"]
    fps_num, fps_den = (int(part) for part in manifest["clip"]["fps"].split("/", 1))
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
        "--arg",
        f"vs_threads={args.vs_threads}",
        VSGAN_SCRIPT,
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


def _validate_parity_engine(
    sidecar: dict[str, Any],
    manifest: dict[str, Any],
    variant_name: str,
    onnx_path: Path,
) -> None:
    validate_static_engine_contract(sidecar, manifest, variant_name, onnx_path)
    if "stronglyTyped" not in sidecar.get("builder_flags", []):
        raise CompetitorError("VSGAN parity engine must be strongly typed")
    version = str(sidecar.get("tensorrt_version", "")).replace(".", "")
    if not version.startswith("1016"):
        raise CompetitorError("Stock VSGAN engine must be built by TensorRT 10.16")


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable stock VSGAN plan."""
    manifest = load_json(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, "vsgan")
    parameters = benchmark_parameters(args, manifest)
    variant = find_variant(manifest, args.variant)
    args.input = str(Path("/app") / variant["path"])
    output_dir = Path(args.output_dir)
    warmup = build_vsgan_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-warmup.mp4",
        frames=parameters["warmup_frames"],
    )
    measured = build_vsgan_command(
        args,
        manifest,
        output_path=output_dir / "dry-run-output.mp4",
        frames=parameters["frames"],
    )
    parameters.update(
        {
            "mode": args.mode,
            "cuda_graph": args.cuda_graph,
            "num_streams": args.num_streams,
            "vspipe_requests": args.requests,
            "vapoursynth_threads": args.vs_threads,
            "batch_size": 1,
            "full_frame": True,
            "tiling": False,
            "bitrate_mbps": variant["benchmark_output"]["bitrate_mbps"],
            "max_compute_processes": 2,
            "max_graphics_processes": 0,
        }
    )
    plan = plan_document(
        product="VSGAN-tensorrt-docker",
        backend="VapourSynth/vstrt",
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
            "Stock VSGAN uses TensorRT 10.16 and therefore receives a separate engine.",
            "The engine must be built from the same canonical ONNX on the same GPU.",
            "The mounted .vpy file is configuration only; no VSGAN source is patched.",
        ],
    )
    return plan, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark stock VSGAN")
    add_common_arguments(parser, engine=True)
    parser.add_argument("--mode", choices=["parity", "tuned"], default="parity")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument("--vs-threads", type=int, default=8)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.requests <= 0 or args.num_streams <= 0 or args.vs_threads <= 0:
            raise CompetitorError("requests, streams and threads must be positive")
        if args.mode == "parity" and (
            args.requests != 1 or args.num_streams != 1 or args.cuda_graph
        ):
            raise CompetitorError(
                "Parity mode requires requests=1, num_streams=1 and CUDA Graph disabled"
            )
        plan, manifest = build_plan(args)
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        engine = Path(args.engine)
        sidecar, sidecar_path = load_engine_contract(engine)
        parameters = plan["parameters"]
        variant = find_variant(manifest, args.variant)
        input_path = Path(args.input)
        onnx_path = Path(find_model_variant(manifest, args.variant)["fp16_path"])
        _validate_parity_engine(sidecar, manifest, args.variant, onnx_path)
        lock_path = Path(manifest["lock_path"])
        build_log = Path(str(sidecar.get("build_log", "")))
        if not build_log.is_file():
            raise CompetitorError(f"VSGAN engine build log not found: {build_log}")
        config = ExternalVideoSuiteConfig(
            product="VSGAN-tensorrt-docker",
            backend="VapourSynth/vstrt",
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
            max_compute_processes=parameters["max_compute_processes"],
            max_graphics_processes=parameters["max_graphics_processes"],
            output_contract=output_contract(
                manifest,
                variant,
                frames=parameters["frames"],
                enforce_bitrate=True,
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
            implementation_parameters={
                "mode": args.mode,
                "requests": args.requests,
                "num_streams": args.num_streams,
                "vapoursynth_threads": args.vs_threads,
                "cuda_graph": args.cuda_graph,
                "batch_size": 1,
                "full_frame": True,
                "tiling": False,
            },
            warmup_command=lambda path, frames: build_vsgan_command(
                args, manifest, output_path=path, frames=frames
            ),
            measured_command=lambda path, frames: build_vsgan_command(
                args, manifest, output_path=path, frames=frames
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
