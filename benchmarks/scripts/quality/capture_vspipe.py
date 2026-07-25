#!/usr/bin/env python3
"""Capture RGBS model input/output tensors from a VapourSynth/vstrt graph."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import collect_image_identity, sha256_file
from ai_media.benchmarking.runner import BenchmarkError, load_engine_contract
from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    TensorArtifact,
    create_tensor_artifact,
    parse_frame_indices,
    write_capture_manifest,
)
from benchmarks.scripts.runners.common import (
    CompetitorError,
    find_model_variant,
    find_variant,
    implementation_config,
    load_json,
    validate_static_engine_contract,
)
from benchmarks.scripts.runners.vapoursynth_profile import (
    VapourSynthExecutionProfile,
    add_execution_profile_arguments,
    comparison_class,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runners.vsgan import _validate_parity_engine
from benchmarks.scripts.workloads.manifest import load_manifest, repo_path


def _quality_frame_indices(
    manifest: dict[str, Any],
    override: str | None,
) -> tuple[int, ...]:
    value = (
        override
        if override is not None
        else ",".join(
            str(index)
            for index in manifest["quality"]["model_space"]["frame_indices"]
        )
    )
    return parse_frame_indices(value, frame_count=int(manifest["clip"]["frames"]))


def build_capture_command(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    frame_index: int,
    stage: str,
    profile: VapourSynthExecutionProfile | None = None,
) -> list[str]:
    """Build one raw RGBS vspipe capture command."""
    profile = profile or resolve_execution_profile(args, args.implementation)
    command = [
        "vspipe",
        "--start",
        str(frame_index),
        "--end",
        str(frame_index),
    ]
    if profile.requests is not None:
        command.extend(["--requests", str(profile.requests)])
    command.extend(
        [
            "--arg",
            f"source={input_path}",
            "--arg",
            f"engine={args.engine}",
            "--arg",
            f"gpu_id={args.gpu_id}",
            "--arg",
            f"cuda_graph={int(profile.cuda_graph)}",
            "--arg",
            f"num_streams={profile.num_streams}",
            "--arg",
            f"model_space_stage={stage}",
        ]
    )
    if profile.vapoursynth_threads is not None:
        command.extend(
            ["--arg", f"vs_threads={profile.vapoursynth_threads}"]
        )
    command.extend([args.script, str(output_path)])
    return command


def _run_capture(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            raise ModelSpaceError(f"Cannot start vspipe: {exc}") from exc
    if result.returncode != 0:
        raise ModelSpaceError(
            f"vspipe model-space capture failed; see {log_path}"
        )


def normalize_vapoursynth_rgbs(
    source_path: Path,
    output_path: Path,
    *,
    shape: tuple[int, int, int],
) -> None:
    """Rewrite VapourSynth's physical GBR plane order as logical RGB CHW."""
    plane_bytes = shape[1] * shape[2] * 4
    expected_size = shape[0] * plane_bytes
    if source_path.stat().st_size != expected_size:
        raise ModelSpaceError(
            "Unexpected raw RGBS size: "
            f"{source_path.stat().st_size} != {expected_size}"
        )

    def copy_plane(source: Any, output: Any, plane_index: int) -> None:
        source.seek(plane_index * plane_bytes)
        remaining = plane_bytes
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ModelSpaceError("Raw RGBS tensor ended inside a plane")
            output.write(chunk)
            remaining -= len(chunk)

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with source_path.open("rb") as source, temporary_path.open("wb") as output:
            for plane_index in (2, 0, 1):
                copy_plane(source, output, plane_index)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def capture(args: argparse.Namespace) -> Path:
    """Capture selected RGBS frames before and after the vstrt model node."""
    profile = resolve_execution_profile(args, args.implementation)
    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, args.implementation)
    validate_declared_profile(implementation, profile)
    clip_variant = find_variant(manifest, args.variant)
    model_variant = find_model_variant(manifest, args.variant)
    input_path = repo_path(root, clip_variant["path"])
    onnx_path = repo_path(root, model_variant["fp16_path"])
    engine_path = Path(args.engine)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = _quality_frame_indices(manifest, args.frame_indices)

    sidecar, _ = load_engine_contract(engine_path)
    if args.implementation == "vsgan":
        _validate_parity_engine(
            sidecar,
            manifest,
            args.variant,
            onnx_path,
            str(implementation["upstream_image"]),
            str(implementation["encoder_ffmpeg_package"]),
        )
    else:
        validate_static_engine_contract(
            sidecar,
            manifest,
            args.variant,
            onnx_path,
        )

    input_shape = (
        3,
        int(model_variant["input_height"]),
        int(model_variant["input_width"]),
    )
    output_shape = (
        3,
        int(clip_variant["benchmark_output"]["height"]),
        int(clip_variant["benchmark_output"]["width"]),
    )
    artifacts: list[TensorArtifact] = []
    for frame_index in frame_indices:
        for stage, shape in (("input", input_shape), ("output", output_shape)):
            output_path = output_dir / f"{stage}.frame-{frame_index:06d}.f32"
            raw_path = output_dir / f"{stage}.frame-{frame_index:06d}.gbr.f32"
            log_path = output_dir / f"{stage}.frame-{frame_index:06d}.log"
            command = build_capture_command(
                args,
                input_path=input_path,
                output_path=raw_path,
                frame_index=frame_index,
                stage=stage,
                profile=profile,
            )
            _run_capture(command, log_path)
            normalize_vapoursynth_rgbs(
                raw_path,
                output_path,
                shape=shape,
            )
            raw_path.unlink()
            artifacts.append(
                create_tensor_artifact(
                    stage=stage,
                    frame_index=frame_index,
                    shape=shape,
                    path=output_path,
                    root=output_dir,
                )
            )

    manifest_path = output_dir / "manifest.json"
    write_capture_manifest(
        manifest_path,
        implementation=(
            "vs-mlrt"
            if args.implementation == "vstrt"
            else "VSGAN-tensorrt-docker"
        ),
        comparison_class=comparison_class(
            str(implementation["comparison_class"]),
            profile,
        ),
        workload_id=manifest["id"],
        variant=args.variant,
        input_sha256=sha256_file(input_path),
        onnx_sha256=sha256_file(onnx_path),
        engine_sha256=sha256_file(engine_path),
        image=collect_image_identity(
            default_reference=str(implementation["image"]),
        ),
        execution_profile=profile.as_parameters(),
        artifacts=artifacts,
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["vstrt", "vsgan"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--implementations",
        default="/app/benchmarks/implementations.json",
    )
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/app")
    parser.add_argument("--gpu-id", type=int, default=0)
    add_execution_profile_arguments(parser)
    parser.add_argument(
        "--frame-indices",
        default=None,
        help="Override canonical comma-separated zero-based frame indices",
    )
    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        manifest_path = capture(args)
    except (
        BenchmarkError,
        CompetitorError,
        ModelSpaceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Model-space capture written: {manifest_path}")


if __name__ == "__main__":
    main()
