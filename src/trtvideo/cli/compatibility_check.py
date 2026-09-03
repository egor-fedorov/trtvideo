"""One-command model compatibility workflow."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from trtvideo.compatibility.workflow import (
    MODEL_TOOLS_IMAGE,
    CompatibilityOptions,
    CompatibilityWorkflowError,
    SourceFormat,
    build_plan,
    run_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the compatibility-check parser."""
    parser = argparse.ArgumentParser(
        prog="trtvideo compatibility-check",
        description=(
            "Export or prepare a model, build an engine, run a live-action smoke test, "
            "and write issue-ready compatibility evidence"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Spandrel-compatible .pth checkpoint")
    source.add_argument("--onnx", type=Path, help="Existing static or dynamic ONNX model")
    parser.add_argument("--model-name", required=True, help="Public model name and version")
    parser.add_argument("--model-source", required=True, help="Public model source URL")
    parser.add_argument("--model-license", required=True, help="SPDX license or public URL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Persistent workspace")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional custom video; default uses the pinned Jacqueville live-action fixture",
    )
    parser.add_argument(
        "--scale",
        type=int,
        help="Scale for a dynamic ONNX graph when metadata cannot provide it",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--resume", action="store_true", help="Resume verified completed steps")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and estimates without creating files or using the GPU",
    )
    parser.add_argument("--verbose", action="store_true", help="Forward verbose logging")
    return parser


def _image_variant() -> str:
    variant = os.environ.get("TRTVIDEO_IMAGE_VARIANT")
    if variant not in {"production", "model-tools", "benchmark", "development"}:
        raise CompatibilityWorkflowError(
            "trtvideo compatibility-check must run in a published trtvideo image"
        )
    return variant


def _require_source_capability(source_format: SourceFormat, variant: str) -> None:
    if source_format == "checkpoint" and variant == "production":
        raise CompatibilityWorkflowError(
            "Checkpoint compatibility requires the model-tools image; run it with "
            f"{MODEL_TOOLS_IMAGE}:vX.Y.Z or pass an existing static ONNX to the "
            "production image"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Plan and run one complete compatibility check."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_format: SourceFormat = "checkpoint" if args.checkpoint is not None else "onnx"
        variant = _image_variant()
        _require_source_capability(source_format, variant)
        source_artifact = args.checkpoint or args.onnx
        assert source_artifact is not None
        options = CompatibilityOptions(
            source_format=source_format,
            source_artifact=source_artifact,
            model_name=args.model_name,
            model_source=args.model_source,
            model_license=args.model_license,
            output_dir=args.output_dir,
            input_video=args.input,
            scale=args.scale,
            gpu_id=args.gpu_id,
            resume=args.resume,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        plan = build_plan(options)
        if variant == "production" and any(step.key == "prepare" for step in plan.steps):
            raise CompatibilityWorkflowError(
                "Dynamic ONNX compatibility requires the model-tools image for graph "
                f"preparation; use {MODEL_TOOLS_IMAGE}:vX.Y.Z or provide a static ONNX"
            )
        run_workflow(plan)
    except CompatibilityWorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
