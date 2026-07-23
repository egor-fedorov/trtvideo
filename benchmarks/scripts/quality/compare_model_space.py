#!/usr/bin/env python3
"""Compare model-space captures against the canonical quality contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    TensorThresholds,
    compare_captures,
)
from benchmarks.scripts.workloads.manifest import load_manifest


def _load_thresholds(manifest: dict[str, Any]) -> dict[str, TensorThresholds]:
    values = manifest["quality"]["model_space"]["thresholds"]
    return {
        stage: TensorThresholds.from_dict(values[stage], stage=stage)
        for stage in ("input", "output")
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Create and persist the model-space parity report."""
    manifest = load_manifest(Path(args.manifest))
    report = compare_captures(
        Path(args.reference),
        [Path(path) for path in args.candidate],
        thresholds=_load_thresholds(manifest),
    )
    if report["workload_id"] != manifest["id"]:
        raise ModelSpaceError(
            "Capture workload does not match the canonical workload manifest"
        )
    if report["variant"] != args.variant:
        raise ModelSpaceError(
            "Capture variant does not match the requested comparison variant"
        )
    canonical_frames = manifest["quality"]["model_space"]["frame_indices"]
    if report["frame_indices"] != canonical_frames:
        raise ModelSpaceError(
            "Capture frame indices do not match the canonical quality contract: "
            f"{report['frame_indices']} != {canonical_frames}"
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
    print(
        f"Model-space parity {report['status']}: {args.output}",
        file=sys.stderr,
    )
    if report["status"] != "valid":
        sys.exit(2)


if __name__ == "__main__":
    main()
