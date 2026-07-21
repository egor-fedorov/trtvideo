#!/usr/bin/env python3
"""Plan or run the canonical product-level Video2X benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai_media.benchmarking.runner import write_summary_target
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

VIDEO2X_BIN = "/usr/bin/video2x"


def build_video2x_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
) -> CommandSpec:
    """Build a stock Video2X 6.4.0 RealESRGAN x2 invocation."""
    variant = find_variant(manifest, args.variant)
    bitrate_bps = int(variant["benchmark_output"]["bitrate_mbps"] * 1_000_000)
    fps_num, fps_den = (int(part) for part in manifest["clip"]["fps"].split("/", 1))
    gop = max(1, round(fps_num / fps_den))
    command = [
        VIDEO2X_BIN,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--processor",
        "realesrgan",
        "--scaling-factor",
        "2",
        "--realesrgan-model",
        args.video2x_model,
        "--device",
        str(args.gpu_id),
        "--hwaccel",
        args.hwaccel,
        "--codec",
        "h264_nvenc",
        "--pix-fmt",
        "yuv420p",
        "--bit-rate",
        str(bitrate_bps),
        "--rc-buffer-size",
        str(2 * bitrate_bps),
        "--rc-min-rate",
        str(bitrate_bps),
        "--rc-max-rate",
        str(bitrate_bps),
        "--gop-size",
        str(gop),
        "--max-b-frames",
        "0",
        "--keyint-min",
        str(gop),
        "--extra-encoder-option",
        "preset=p4",
        "tune=hq",
        "rc=cbr",
        "--no-copy-streams",
        "--no-progress",
        "--log-level",
        "warn",
    ]
    return command_spec(command)


def build_trim_command(
    manifest: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    frames: int,
) -> list[str]:
    """Build an untimed deterministic clip used for Video2X warmup or smoke."""
    encode = manifest["clip"]["encode"]
    x264_params = (
        f"keyint={encode['gop_frames']}:min-keyint={encode['gop_frames']}:"
        f"scenecut=0:bframes={encode['b_frames']}:"
        f"colorprim={encode['color_primaries']}:transfer={encode['color_transfer']}:"
        f"colormatrix={encode['color_space']}:range=limited"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-frames:v",
        str(frames),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        encode["codec"],
        "-preset",
        encode["preset"],
        "-crf",
        str(encode["crf"]),
        "-pix_fmt",
        encode["pixel_format"],
        "-x264-params",
        x264_params,
        "-color_range",
        encode["color_range"],
        "-colorspace",
        encode["color_space"],
        "-color_trc",
        encode["color_transfer"],
        "-color_primaries",
        encode["color_primaries"],
        str(output_path),
    ]


def _prepare_clip(command: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise CompetitorError(f"Failed to prepare Video2X clip: {detail}")


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a machine-readable Video2X plan and return the workload manifest."""
    manifest = load_json(Path(args.manifest))
    competitors = load_json(Path(args.competitors))
    implementation = competitor_config(competitors, "video2x")
    parameters = benchmark_parameters(args, manifest)
    variant = find_variant(manifest, args.variant)
    canonical_input = Path(variant["path"])
    prepared_dir = Path(args.output_dir) / "prepared-inputs"
    warmup_input = prepared_dir / f"warmup-{parameters['warmup_frames']}.mp4"
    measured_input = (
        canonical_input
        if parameters["frames"] == manifest["clip"]["frames"]
        else prepared_dir / f"measured-{parameters['frames']}.mp4"
    )
    warmup_trim = build_trim_command(
        manifest,
        input_path=canonical_input,
        output_path=warmup_input,
        frames=parameters["warmup_frames"],
    )
    measured_trim = (
        None
        if measured_input == canonical_input
        else build_trim_command(
            manifest,
            input_path=canonical_input,
            output_path=measured_input,
            frames=parameters["frames"],
        )
    )
    warmup = build_video2x_command(
        args,
        manifest,
        input_path=warmup_input,
        output_path=Path(args.output_dir) / "dry-run-warmup.mp4",
    )
    measured = build_video2x_command(
        args,
        manifest,
        input_path=measured_input,
        output_path=Path(args.output_dir) / "dry-run-output.mp4",
    )
    parameters.update(
        {
            "model": args.video2x_model,
            "scale": 2,
            "hwaccel": args.hwaccel,
            "bitrate_mbps": variant["benchmark_output"]["bitrate_mbps"],
            "canonical_measured_input": measured_input == canonical_input,
        }
    )
    setup_commands = [warmup_trim] + ([measured_trim] if measured_trim else [])
    plan = plan_document(
        product="Video2X",
        backend="realesrgan-ncnn-vulkan",
        comparison_class=implementation["comparison_class"],
        implementation=implementation,
        manifest=manifest,
        variant_name=args.variant,
        parameters=parameters,
        commands={
            "setup": setup_commands,
            "warmup": warmup,
            "measured": measured,
            "warmup_display": display_command(warmup),
            "measured_display": display_command(measured),
        },
        assets=[
            asset_requirement(str(canonical_input), "input"),
            asset_requirement(args.manifest, "workload_manifest"),
        ],
        limitations=[
            "Stock Video2X does not support RealESRGAN_x2plus or TensorRT.",
            "The bundled realesr-animevideov3 x2 NCNN/Vulkan model makes this product-level only.",
            "A non-default --frames value uses an untimed re-encoded input and is smoke-only.",
        ],
    )
    plan["prepared_inputs"] = {
        "canonical": str(canonical_input),
        "warmup": str(warmup_input),
        "measured": str(measured_input),
    }
    return plan, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Video2X RealESRGAN")
    add_common_arguments(parser, engine=False)
    parser.add_argument("--video2x-model", default="realesr-animevideov3")
    parser.add_argument(
        "--hwaccel",
        default="none",
        help="Video2X decode acceleration (stock RealESRGAN requires software frames)",
    )
    parser.add_argument("--keep-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        plan, manifest = build_plan(args)
        if args.video2x_model != plan["implementation"]["model"]:
            raise CompetitorError(
                "Changing the Video2X model invalidates the pinned product-level contract"
            )
        if args.dry_run:
            write_json_target(plan, args.json)
            return

        prepared = plan["prepared_inputs"]
        setup_commands = plan["commands"]["setup"]
        for command in setup_commands:
            _prepare_clip(command, Path(command[-1]))
        parameters = plan["parameters"]
        variant = find_variant(manifest, args.variant)
        canonical_input = Path(prepared["canonical"])
        warmup_input = Path(prepared["warmup"])
        measured_input = Path(prepared["measured"])
        config = ExternalVideoSuiteConfig(
            product="Video2X",
            backend="realesrgan-ncnn-vulkan",
            comparison_class=plan["comparison_class"],
            implementation=plan["implementation"],
            workload_id=manifest["id"],
            variant=args.variant,
            input_path=canonical_input,
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
                "input": canonical_input,
                "workload_manifest": Path(args.manifest),
            },
            warmup_command=lambda path, _frames: build_video2x_command(
                args,
                manifest,
                input_path=warmup_input,
                output_path=path,
            ),
            measured_command=lambda path, _frames: build_video2x_command(
                args,
                manifest,
                input_path=measured_input,
                output_path=path,
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
