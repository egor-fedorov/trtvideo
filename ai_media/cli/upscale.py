"""Unified video upscale CLI."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Create parser for all current upscale backends."""
    parser = argparse.ArgumentParser(prog="upscale", description="TensorRT media enhancer")
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "nvcodec"],
        default="nvcodec",
        help="Video IO backend",
    )
    engine_source = parser.add_mutually_exclusive_group(required=True)
    engine_source.add_argument("--engine", help="Path to .engine file")
    engine_source.add_argument(
        "--model",
        help="Path to model registry directory, registry manifest, or engine manifest",
    )
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--output", default=None, help="Output video")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument(
        "--engine-precision",
        choices=["fp16", "fp32"],
        default=None,
        help="Preferred precision when selecting an engine from --model",
    )
    parser.add_argument(
        "--engine-io-precision",
        choices=["fp16", "fp32"],
        default=None,
        help="Preferred input/output binding precision when selecting an engine from --model",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Frames to exclude from profiling/benchmark summaries",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N frames")
    parser.add_argument("--profile", action="store_true", help="Per-stage profiling")
    parser.add_argument("--profile-json", default=None, help="Write profiling JSON summary")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Experimental: capture TensorRT enqueue with CUDA Graph",
    )
    parser.add_argument("--crf", type=int, default=18, help="Encoding quality")
    parser.add_argument(
        "--bitrate-mbps",
        type=float,
        default=None,
        help=(
            "Explicit NVENC target bitrate in Mbps; "
            "default nvcodec mode estimates it from source bitrate"
        ),
    )
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc"], help="NVENC codec")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "ffmpeg":
        from ai_media.pipelines.ffmpeg import FfmpegPipeline

        FfmpegPipeline(args).run()
        return
    if args.backend == "nvcodec":
        from ai_media.pipelines.nvcodec import NvcodecPipeline

        NvcodecPipeline(args).run()
        return

    print(f"ERROR: unsupported backend: {args.backend}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
