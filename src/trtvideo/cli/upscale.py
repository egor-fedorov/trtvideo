"""GPU-resident video upscale CLI."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the NVDEC/CV-CUDA/TensorRT/NVENC upscale parser."""
    parser = argparse.ArgumentParser(prog="upscale", description="TensorRT video upscaler")
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


def main() -> None:
    args = build_parser().parse_args()
    from trtvideo.pipelines.nvcodec import NvcodecPipeline

    NvcodecPipeline(args).run()


if __name__ == "__main__":
    main()
