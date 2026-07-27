"""Internal CLI for the self-contained ``make demo`` workflow."""

import argparse
import sys
from pathlib import Path

from trtvideo.demo import DemoError
from trtvideo.demo.workflow import run_demo


def build_parser() -> argparse.ArgumentParser:
    """Create the internal demo workflow parser."""
    parser = argparse.ArgumentParser(description="Run the self-contained GPU upscale demo")
    parser.add_argument("--root", type=Path, default=Path("/app/.demo"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = run_demo(args.root, gpu_id=args.gpu_id, force=args.force)
    except DemoError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print("\nDemo completed and validated.")
    print(f"Output: {paths.output_video}")
    print(f"Report: {paths.report}")


if __name__ == "__main__":
    main()
