#!/usr/bin/env python3
"""Compare TensorRT outputs produced from one exact shared RGB input tensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    TensorThresholds,
    compare_inference_captures,
    validate_report_scope,
)
from benchmarks.scripts.workloads.manifest import load_manifest


def _output_thresholds(manifest: dict[str, Any]) -> TensorThresholds:
    values = manifest["quality"]["model_space"]["inference"]["output_thresholds"]
    return TensorThresholds.from_dict(values, stage="output")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Create and persist the shared-input inference parity report."""
    manifest = load_manifest(Path(args.manifest))
    report = compare_inference_captures(
        Path(args.reference),
        [Path(path) for path in args.candidate],
        output_thresholds=_output_thresholds(manifest),
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
    print(f"Inference parity {report['status']}: {args.output}", file=sys.stderr)
    if report["status"] != "valid":
        sys.exit(2)


if __name__ == "__main__":
    main()
