"""Prepare a deterministic live-action input for model compatibility checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from trtvideo.compatibility.input import (
    DEFAULT_INPUT_FRAMES,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    CompatibilityInputError,
    InputPreparation,
    prepare_input,
)


def _size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().replace("*", "x").split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    """Create the low-level input preparation parser."""
    parser = argparse.ArgumentParser(
        prog="prepare-compatibility-input",
        description="Prepare and validate a 120-frame SDR H.264 compatibility input",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional user video; default downloads the pinned Jacqueville source",
    )
    parser.add_argument("--output", required=True, type=Path, help="Prepared MP4 path")
    parser.add_argument("--manifest", type=Path, help="Manifest path (default: OUTPUT.input.json)")
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Pinned source cache (required when --input is omitted)",
    )
    parser.add_argument(
        "--size",
        type=_size,
        default=(DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT),
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_INPUT_FRAMES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one input or report a contract failure."""
    args = build_parser().parse_args(argv)
    width, height = args.size
    manifest = args.manifest or Path(f"{args.output}.input.json")
    source_cache = args.source_cache
    if args.input is None and source_cache is None:
        source_cache = args.output.parent / "Jacqueville-beach-2026.webm"
    try:
        prepare_input(
            InputPreparation(
                output=args.output,
                manifest=manifest,
                width=width,
                height=height,
                frames=args.frames,
                source=args.input,
                source_cache=source_cache,
            )
        )
    except CompatibilityInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Compatibility input valid: {args.output}")
    print(f"Input manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
