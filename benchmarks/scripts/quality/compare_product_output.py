#!/usr/bin/env python3
"""Compare retained benchmark MP4 outputs with PSNR, SSIM, and visual crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.scripts.quality.product_output import (
    ProductOutputError,
    compare_product_outputs,
)
from benchmarks.scripts.workloads.manifest import find_clip_variant, load_manifest


def compare(args: argparse.Namespace) -> dict:
    """Run the product-output gate and persist its report."""
    root = Path(args.root).resolve()
    workload = load_manifest(Path(args.manifest))
    variant = find_clip_variant(workload, args.variant)
    output_path = Path(args.output)
    report = compare_product_outputs(
        Path(args.reference),
        [Path(path) for path in args.candidate],
        workload=workload,
        variant=variant,
        root=root,
        output_dir=output_path.parent,
    )
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/app")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = compare(args)
    except (ProductOutputError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Product-output parity {report['status']}: {args.output}", file=sys.stderr)
    if report["status"] != "valid":
        sys.exit(2)


if __name__ == "__main__":
    main()
