#!/usr/bin/env python3
"""Measure production RGB preprocessing differences without gating quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    compare_preprocessing_captures,
    validate_report_scope,
)
from benchmarks.scripts.workloads.manifest import load_manifest


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Create and persist the production preprocessing diagnostic."""
    manifest = load_manifest(Path(args.manifest))
    report = compare_preprocessing_captures(
        Path(args.reference),
        [Path(path) for path in args.candidate],
    )
    report["contract_version"] = manifest["quality"]["model_space"]["contract_version"]
    validate_report_scope(
        report,
        workload_id=manifest["id"],
        variant=args.variant,
        frame_indices=manifest["quality"]["model_space"]["frame_indices"],
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        report = compare(args)
    except (ModelSpaceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Preprocessing diagnostic {report['status']}: {args.output}", file=sys.stderr)
    if report["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    main()
