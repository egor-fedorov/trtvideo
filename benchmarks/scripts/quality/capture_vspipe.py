#!/usr/bin/env python3
"""Capture RGBS model input/output tensors from a VapourSynth/vstrt graph."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.benchmark import (
    CompetitorError,
    implementation_config,
    load_json,
)
from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
    validate_static_engine_contract,
    validate_vsgan_engine_contract,
)
from benchmarks.scripts.quality.model_space import (
    CaptureManifest,
    ModelSpaceError,
    TensorArtifact,
    create_tensor_artifact,
    parse_frame_indices,
    write_capture_manifest,
)
from benchmarks.scripts.runners.vapoursynth_profile import (
    VapourSynthExecutionProfile,
    add_execution_profile_arguments,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runtime.environment import collect_image_identity, sha256_file
from benchmarks.scripts.workloads.manifest import (
    WorkloadError,
    find_clip_variant,
    find_model_variant,
    load_manifest,
    repo_path,
)


def _quality_frame_indices(
    manifest: dict[str, Any],
    override: str | None,
) -> tuple[int, ...]:
    value = (
        override
        if override is not None
        else ",".join(str(index) for index in manifest["quality"]["model_space"]["frame_indices"])
    )
    return parse_frame_indices(value, frame_count=int(manifest["clip"]["frames"]))


def build_capture_command(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    frame_index: int,
    stage: str,
    shared_input: bool = False,
    profile: VapourSynthExecutionProfile | None = None,
) -> list[str]:
    """Build one raw RGBS vspipe capture command."""
    profile = profile or resolve_execution_profile(args, args.implementation)
    command = [
        "vspipe",
        "--start",
        "0" if shared_input else str(frame_index),
        "--end",
        "0" if shared_input else str(frame_index),
    ]
    if profile.requests is not None:
        command.extend(["--requests", str(profile.requests)])
    command.extend(
        [
            "--arg",
            f"{'model_input' if shared_input else 'source'}={input_path}",
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
        command.extend(["--arg", f"vs_threads={profile.vapoursynth_threads}"])
    command.extend([args.script, str(output_path)])
    return command


def _run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            raise ModelSpaceError(f"Cannot start capture command {command[0]}: {exc}") from exc
    if result.returncode != 0:
        raise ModelSpaceError(f"Capture command {command[0]} failed; see {log_path}")


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
            f"Unexpected raw RGBS size: {source_path.stat().st_size} != {expected_size}"
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


def serialize_vapoursynth_rgbs(
    source_path: Path,
    output_path: Path,
    *,
    shape: tuple[int, int, int],
) -> None:
    """Rewrite logical RGB CHW as VapourSynth/FFmpeg physical GBR planes."""
    plane_bytes = shape[1] * shape[2] * 4
    expected_size = shape[0] * plane_bytes
    if source_path.stat().st_size != expected_size:
        raise ModelSpaceError(
            f"Unexpected canonical RGB tensor size: {source_path.stat().st_size} != {expected_size}"
        )

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with source_path.open("rb") as source, temporary_path.open("wb") as output:
            for plane_index in (1, 2, 0):
                source.seek(plane_index * plane_bytes)
                remaining = plane_bytes
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ModelSpaceError("Canonical RGB tensor ended inside a plane")
                    output.write(chunk)
                    remaining -= len(chunk)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _package_shared_input(
    source_path: Path,
    output_path: Path,
    *,
    shape: tuple[int, int, int],
    log_path: Path,
) -> None:
    """Pack one exact RGBS tensor in a lossless one-frame NUT container."""
    raw_path = output_path.with_suffix(".gbr.f32")
    serialize_vapoursynth_rgbs(source_path, raw_path, shape=shape)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "gbrpf32le",
        "-video_size",
        f"{shape[2]}x{shape[1]}",
        "-framerate",
        "1",
        "-i",
        str(raw_path),
        "-frames:v",
        "1",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-f",
        "nut",
        str(output_path),
    ]
    try:
        _run_command(command, log_path)
    finally:
        raw_path.unlink(missing_ok=True)


def capture(args: argparse.Namespace) -> Path:
    """Capture production preprocessing or shared-input vstrt inference."""
    profile = resolve_execution_profile(args, args.implementation)
    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    implementations = load_json(Path(args.implementations))
    implementation = implementation_config(implementations, args.implementation)
    validate_declared_profile(implementation, profile)
    clip_variant = find_clip_variant(manifest, args.variant)
    model_variant = find_model_variant(manifest, args.variant)
    input_path = repo_path(root, clip_variant["path"])
    onnx_path = repo_path(root, model_variant["fp16_path"])
    engine_path = Path(args.engine)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = _quality_frame_indices(manifest, args.frame_indices)
    canonical_manifest_path = (
        Path(args.shared_input_manifest) if args.shared_input_manifest else None
    )
    canonical_capture = (
        CaptureManifest.load(canonical_manifest_path)
        if canonical_manifest_path is not None
        else None
    )

    sidecar, _ = load_engine_contract(engine_path)
    if args.implementation == "vsgan":
        validate_vsgan_engine_contract(
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
    if canonical_capture is not None:
        assert canonical_manifest_path is not None
        if canonical_capture.capture_scope != "production-reference":
            raise ModelSpaceError("Shared inference input must be a production reference")
        if (
            canonical_capture.workload_id != manifest["id"]
            or canonical_capture.variant != args.variant
            or canonical_capture.input_sha256 != sha256_file(input_path)
            or canonical_capture.onnx_sha256 != sha256_file(onnx_path)
        ):
            raise ModelSpaceError("Canonical input capture does not match this workload")
        canonical_inputs = {
            artifact.frame_index: artifact
            for artifact in canonical_capture.artifacts
            if artifact.stage == "input"
        }
        if set(canonical_inputs) != set(frame_indices):
            raise ModelSpaceError("Canonical input capture has the wrong frame set")
        for frame_index in frame_indices:
            source_artifact = canonical_inputs[frame_index]
            if source_artifact.shape != input_shape:
                raise ModelSpaceError(
                    "Canonical input tensor shape does not match this model: "
                    f"{source_artifact.shape} != {input_shape}"
                )
            source_tensor = canonical_manifest_path.parent / source_artifact.path
            packed_input = output_dir / f"input.frame-{frame_index:06d}.nut"
            _package_shared_input(
                source_tensor,
                packed_input,
                shape=input_shape,
                log_path=output_dir / f"input.frame-{frame_index:06d}.ffmpeg.log",
            )
            try:
                for stage, shape in (("input", input_shape), ("output", output_shape)):
                    output_path = output_dir / f"{stage}.frame-{frame_index:06d}.f32"
                    raw_path = output_dir / f"{stage}.frame-{frame_index:06d}.gbr.f32"
                    log_path = output_dir / f"{stage}.frame-{frame_index:06d}.log"
                    command = build_capture_command(
                        args,
                        input_path=packed_input,
                        output_path=raw_path,
                        frame_index=frame_index,
                        stage=stage,
                        shared_input=True,
                        profile=profile,
                    )
                    try:
                        _run_command(command, log_path)
                        normalize_vapoursynth_rgbs(raw_path, output_path, shape=shape)
                    finally:
                        raw_path.unlink(missing_ok=True)
                    artifacts.append(
                        create_tensor_artifact(
                            stage=stage,
                            frame_index=frame_index,
                            shape=shape,
                            path=output_path,
                            root=output_dir,
                        )
                    )
            finally:
                packed_input.unlink(missing_ok=True)
    else:
        for frame_index in frame_indices:
            stage = "input"
            shape = input_shape
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
            _run_command(command, log_path)
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
        implementation=("vs-mlrt" if args.implementation == "vstrt" else "VSGAN-tensorrt-docker"),
        capture_scope=(
            "shared-input-inference"
            if canonical_manifest_path is not None
            else "production-preprocessing"
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
        canonical_input_manifest_sha256=(
            sha256_file(canonical_manifest_path) if canonical_manifest_path is not None else None
        ),
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
    parser.add_argument(
        "--shared-input-manifest",
        default=None,
        help="Inject input tensors from a trtvideo production capture",
    )
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
        CompetitorError,
        EngineContractError,
        ModelSpaceError,
        OSError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Tensor capture written: {manifest_path}")


if __name__ == "__main__":
    main()
