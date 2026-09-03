"""Model compatibility evidence bundle CLI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from trtvideo.compatibility.evidence import CompatibilityEvidenceError
from trtvideo.compatibility.report import CompatibilityRequest, generate_compatibility_report


def build_parser() -> argparse.ArgumentParser:
    """Create the compatibility-report parser."""
    parser = argparse.ArgumentParser(
        prog="trtvideo compatibility-report",
        description=(
            "Collect existing model, engine, environment, command, and smoke-test evidence "
            "into sanitized JSON and issue-ready Markdown"
        ),
    )
    parser.add_argument("--model-name", required=True, help="Public model name and version")
    parser.add_argument("--model-source", required=True, help="Public model source URL")
    parser.add_argument("--model-license", required=True, help="SPDX license or public URL")
    parser.add_argument(
        "--source-format",
        required=True,
        choices=["checkpoint", "onnx"],
        help="Identity represented by --source-artifact",
    )
    parser.add_argument(
        "--source-artifact",
        required=True,
        type=Path,
        help="Exact checkpoint or source ONNX to hash",
    )
    parser.add_argument("--engine", required=True, type=Path, help="Static TensorRT engine")
    parser.add_argument("--input", required=True, type=Path, help="Smoke-test input video")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Optional prepare-compatibility-input manifest to bind into the report",
    )
    parser.add_argument(
        "--processed-output",
        required=True,
        type=Path,
        help="Completed smoke-test output video",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=120,
        help="Expected smoke-test frame count (default: 120)",
    )
    parser.add_argument(
        "--commands-file",
        required=True,
        type=Path,
        help="UTF-8 log containing exact preparation and runtime commands",
    )
    parser.add_argument(
        "--export-conformance",
        type=Path,
        default=None,
        help="Generated export-conformance JSON (required for checkpoints)",
    )
    parser.add_argument(
        "--image-reference",
        default=os.environ.get("TRTVIDEO_IMAGE_REF", "unknown"),
        help="Exact image tag or digest (default: TRTVIDEO_IMAGE_REF)",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for JSON and issue Markdown",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and summarize a compatibility evidence bundle."""
    args = build_parser().parse_args(argv)
    request = CompatibilityRequest(
        model_name=args.model_name,
        model_source=args.model_source,
        model_license=args.model_license,
        source_format=args.source_format,
        source_artifact=args.source_artifact,
        engine=args.engine,
        input_video=args.input,
        output_video=args.processed_output,
        expected_frames=args.expected_frames,
        commands_file=args.commands_file,
        export_conformance=args.export_conformance,
        image_reference=args.image_reference,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        input_manifest=args.input_manifest,
    )
    try:
        report = generate_compatibility_report(request)
    except CompatibilityEvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"ERROR: Cannot write compatibility report ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2
    files = report["written_files"]
    print(f"Compatibility report {report['status']}: {files['json']}")
    print(f"Issue body: {files['markdown']}")
    if report["status"] != "valid":
        for error in report["errors"]:
            print(f"  - {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
