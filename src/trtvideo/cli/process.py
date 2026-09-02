"""GPU-resident TensorRT video processing CLI."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trtvideo.pipelines.config import PipelineError, ProcessConfig, default_output_path


def build_parser() -> argparse.ArgumentParser:
    """Create the NVDEC/CV-CUDA/TensorRT/NVENC processing parser."""
    parser = argparse.ArgumentParser(
        prog="trtvideo",
        description="GPU-resident TensorRT video processing",
        epilog=(
            "Run 'trtvideo doctor' to check the static runtime environment or "
            "'trtvideo compatibility-report --help' to prepare model evidence."
        ),
    )
    parser.add_argument("--engine", required=True, help="Path to .engine file")
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--output", default=None, help="Output video")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Frames to exclude from profiling/benchmark summaries",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N frames")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print isolated stage timings; serializes the pipeline per frame, "
            "so timings are not additive and FPS is not throughput"
        ),
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help=(
            "Write isolated stage timings as JSON; serializes the pipeline per "
            "frame and is not a throughput measurement"
        ),
    )
    parser.add_argument(
        "--benchmark-lifecycle-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bitrate-mbps",
        type=float,
        default=None,
        help=(
            "Explicit NVENC target bitrate in Mbps; the default estimates it from source bitrate"
        ),
    )
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc"], help="NVENC codec")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    return parser


def process_config_from_args(args: argparse.Namespace) -> ProcessConfig:
    """Translate CLI parsing concerns into the typed pipeline contract."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else default_output_path(input_path)
    return ProcessConfig(
        engine_path=Path(args.engine),
        input_path=input_path,
        output_path=output_path,
        gpu_id=args.gpu_id,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        log_interval=args.log_interval,
        profile=args.profile,
        profile_json_path=Path(args.profile_json) if args.profile_json is not None else None,
        benchmark_lifecycle_path=(
            Path(args.benchmark_lifecycle_json)
            if args.benchmark_lifecycle_json is not None
            else None
        ),
        bitrate_mbps=args.bitrate_mbps,
        codec=cast(Literal["h264", "hevc"], args.codec),
        verbose=args.verbose,
        quiet=args.quiet,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    from trtvideo.pipelines.nvcodec import NvcodecPipeline

    try:
        NvcodecPipeline(process_config_from_args(args)).run()
    except PipelineError as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
