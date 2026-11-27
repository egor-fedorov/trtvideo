"""Run the self-contained GPU demo from the model-tools image."""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from trtvideo.demo import DemoError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtvideo demo",
        description="Upscale a five-second sample with RealESRGAN in the model-tools image.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".demo"),
        help="Cache and output directory (default: .demo)",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="Visible CUDA GPU index (default: 0)")
    parser.add_argument("--force", action="store_true", help="Rebuild generated assets and engine")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    if not all(shutil.which(tool) for tool in ("export-onnx", "prepare-onnx")):
        print(
            "The demo requires the ghcr.io/egor-fedorov/trtvideo-model-tools image.",
            file=sys.stderr,
        )
        return 2

    from trtvideo.demo.workflow import run_demo

    try:
        paths = run_demo(args.root, gpu_id=args.gpu_id, force=args.force)
    except (DemoError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDemo completed and validated.")
    print(f"Output: {paths.output_video}")
    print(f"Report: {paths.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
