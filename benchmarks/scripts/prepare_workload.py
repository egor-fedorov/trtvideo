#!/usr/bin/env python3
"""Prepare and verify the canonical upscale benchmark workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_STEP = 256 * 1024 * 1024


class WorkloadError(RuntimeError):
    """Raised when workload preparation or validation fails."""


def sha256_file(path: Path) -> str:
    """Return a file SHA256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a workload manifest."""
    try:
        with path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"Cannot read workload manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise WorkloadError("Workload manifest root must be an object")
    validate_manifest(manifest)
    return manifest


def _require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise WorkloadError(f"Manifest field '{key}' must be an object")
    return value


def _require_list(parent: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise WorkloadError(f"Manifest field '{key}' must be a non-empty object array")
    return value


def _validate_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise WorkloadError(f"Manifest field '{field}' must be a relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise WorkloadError(f"Manifest field '{field}' must stay inside the repository")


def _validate_source(source: dict[str, Any], field: str) -> None:
    url = source.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise WorkloadError(f"Manifest field '{field}.url' must be an HTTPS URL")
    checksum = source.get("sha256")
    if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
        raise WorkloadError(f"Manifest field '{field}.sha256' must be a lowercase SHA256")
    if not isinstance(source.get("size_bytes"), int) or source["size_bytes"] <= 0:
        raise WorkloadError(f"Manifest field '{field}.size_bytes' must be positive")
    for metadata_key in ("license_reference", "attribution"):
        if not isinstance(source.get(metadata_key), str) or not source[metadata_key]:
            raise WorkloadError(f"Manifest field '{field}.{metadata_key}' is required")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the Stage 0 workload schema and path boundaries."""
    if manifest.get("schema_version") != 1:
        raise WorkloadError("Unsupported workload schema_version")
    if not isinstance(manifest.get("id"), str) or not manifest["id"]:
        raise WorkloadError("Manifest field 'id' is required")
    _validate_relative_path(manifest.get("lock_path"), "lock_path")
    benchmark = _require_dict(manifest, "benchmark")
    required_benchmark_fields = {
        "warmup_frames",
        "measured_frames",
        "initial_runs",
        "extra_runs_on_spread",
        "spread_threshold",
        "idle_seconds",
        "nvml_sample_interval_ms",
    }
    missing_benchmark = sorted(required_benchmark_fields - benchmark.keys())
    if missing_benchmark:
        raise WorkloadError(
            f"Manifest benchmark fields are missing: {', '.join(missing_benchmark)}"
        )
    for field in (
        "warmup_frames",
        "measured_frames",
        "initial_runs",
        "nvml_sample_interval_ms",
    ):
        if not isinstance(benchmark.get(field), int) or benchmark[field] <= 0:
            raise WorkloadError(f"Manifest field 'benchmark.{field}' must be positive")
    if not isinstance(benchmark.get("extra_runs_on_spread"), int) or (
        benchmark["extra_runs_on_spread"] < 0
    ):
        raise WorkloadError("Manifest field 'benchmark.extra_runs_on_spread' must be non-negative")
    if not isinstance(benchmark.get("spread_threshold"), (int, float)) or not (
        0 <= benchmark["spread_threshold"] < 1
    ):
        raise WorkloadError("Manifest field 'benchmark.spread_threshold' must be in [0, 1)")
    if not isinstance(benchmark.get("idle_seconds"), (int, float)) or (
        benchmark["idle_seconds"] < 0
    ):
        raise WorkloadError("Manifest field 'benchmark.idle_seconds' must be non-negative")

    model = _require_dict(manifest, "model")
    _validate_source(_require_dict(model, "source"), "model.source")
    _validate_relative_path(model.get("weights_path"), "model.weights_path")
    _validate_relative_path(model.get("onnx_dir"), "model.onnx_dir")
    model_variants = _require_list(model, "variants")

    clip = _require_dict(manifest, "clip")
    _validate_source(_require_dict(clip, "source"), "clip.source")
    _validate_relative_path(clip.get("source_path"), "clip.source_path")
    if clip.get("frames") != 1000:
        raise WorkloadError("Canonical workload must contain exactly 1000 frames")
    try:
        fps = Fraction(str(clip.get("fps")))
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkloadError("Manifest field 'clip.fps' must be a rational FPS") from exc
    if fps <= 0:
        raise WorkloadError("Manifest field 'clip.fps' must be positive")
    encode = _require_dict(clip, "encode")
    required_encode_fields = {
        "codec",
        "preset",
        "crf",
        "pixel_format",
        "gop_frames",
        "b_frames",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
    }
    missing = sorted(required_encode_fields - encode.keys())
    if missing:
        raise WorkloadError(f"Manifest clip.encode fields are missing: {', '.join(missing)}")
    clip_variants = _require_list(clip, "variants")

    model_names = set()
    for variant in model_variants:
        name = variant.get("name")
        if not isinstance(name, str) or not name or name in model_names:
            raise WorkloadError("Model variant names must be unique non-empty strings")
        model_names.add(name)
        for dimension in ("input_width", "input_height"):
            if not isinstance(variant.get(dimension), int) or variant[dimension] <= 0:
                raise WorkloadError(f"Model variant '{name}' has invalid {dimension}")
        _validate_relative_path(variant.get("fp32_path"), f"model.{name}.fp32_path")
        _validate_relative_path(variant.get("fp16_path"), f"model.{name}.fp16_path")

    clip_names = set()
    for variant in clip_variants:
        name = variant.get("name")
        if not isinstance(name, str) or not name or name in clip_names:
            raise WorkloadError("Clip variant names must be unique non-empty strings")
        clip_names.add(name)
        for dimension in ("width", "height"):
            if not isinstance(variant.get(dimension), int) or variant[dimension] <= 0:
                raise WorkloadError(f"Clip variant '{name}' has invalid {dimension}")
        _validate_relative_path(variant.get("path"), f"clip.{name}.path")
        benchmark_output = _require_dict(variant, "benchmark_output")
        for dimension in ("width", "height"):
            if (
                not isinstance(benchmark_output.get(dimension), int)
                or benchmark_output[dimension] <= 0
            ):
                raise WorkloadError(
                    f"Clip variant '{name}' has invalid benchmark_output.{dimension}"
                )
        if not isinstance(benchmark_output.get("bitrate_mbps"), (int, float)) or (
            benchmark_output["bitrate_mbps"] <= 0
        ):
            raise WorkloadError(
                f"Clip variant '{name}' has invalid benchmark_output.bitrate_mbps"
            )

    if model_names != clip_names:
        raise WorkloadError("Model and clip variants must use the same names")


def repo_path(root: Path, relative_path: str) -> Path:
    """Resolve a validated repository-relative path."""
    root = root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise WorkloadError(f"Path escapes repository root: {relative_path}")
    return resolved


def verify_source_file(path: Path, source: dict[str, Any]) -> str:
    """Verify source size and checksum and return the checksum."""
    if not path.is_file():
        raise WorkloadError(f"Required file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != source["size_bytes"]:
        raise WorkloadError(
            f"Unexpected size for {path}: expected {source['size_bytes']}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != source["sha256"]:
        raise WorkloadError(
            f"Checksum mismatch for {path}: expected {source['sha256']}, got {actual_hash}"
        )
    return actual_hash


def download_source(source: dict[str, Any], destination: Path, *, force: bool) -> None:
    """Download a source with HTTP range resume and verify it."""
    if destination.exists():
        try:
            verify_source_file(destination, source)
        except WorkloadError:
            if not force:
                raise
            destination.unlink()
        else:
            print(f"Using verified source: {destination}")
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if partial.exists() and partial.stat().st_size >= source["size_bytes"]:
        try:
            verify_source_file(partial, source)
        except WorkloadError:
            if not force:
                raise
            partial.unlink()
        else:
            os.replace(partial, destination)
            print(f"Promoted verified partial download: {destination}")
            return

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ai-media-enhancer-benchmark/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming {destination.name} at {offset} bytes")
    else:
        print(f"Downloading {source['url']}")

    request = urllib.request.Request(source["url"], headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except OSError as exc:
        raise WorkloadError(f"Download failed for {source['url']}: {exc}") from exc

    status = getattr(response, "status", response.getcode())
    append = bool(offset and status == 206)
    if offset and not append:
        print("Server ignored Range request; restarting download")
        offset = 0
    mode = "ab" if append else "wb"
    downloaded = offset
    next_progress = ((downloaded // PROGRESS_STEP) + 1) * PROGRESS_STEP
    try:
        with response, partial.open(mode) as output:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress:
                    print(f"  {downloaded / (1024**3):.2f} GiB downloaded")
                    next_progress += PROGRESS_STEP
    except OSError as exc:
        raise WorkloadError(f"Download interrupted for {source['url']}: {exc}") from exc

    verify_source_file(partial, source)
    os.replace(partial, destination)
    print(f"Verified source: {destination}")


def build_ffmpeg_command(
    manifest: dict[str, Any],
    source: Path,
    variant: dict[str, Any],
    output: Path,
) -> list[str]:
    """Build the deterministic input clip preparation command."""
    clip = manifest["clip"]
    encode = clip["encode"]
    x264_params = (
        f"keyint={encode['gop_frames']}:min-keyint={encode['gop_frames']}:"
        f"scenecut=0:bframes={encode['b_frames']}:"
        f"colorprim={encode['color_primaries']}:transfer={encode['color_transfer']}:"
        f"colormatrix={encode['color_space']}:range=limited"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-frames:v",
        str(clip["frames"]),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={variant['width']}:{variant['height']}:flags=lanczos,setsar=1",
        "-r",
        clip["fps"],
        "-c:v",
        encode["codec"],
        "-preset",
        encode["preset"],
        "-crf",
        str(encode["crf"]),
        "-pix_fmt",
        encode["pixel_format"],
        "-x264-params",
        x264_params,
        "-color_range",
        encode["color_range"],
        "-colorspace",
        encode["color_space"],
        "-color_trc",
        encode["color_transfer"],
        "-color_primaries",
        encode["color_primaries"],
        "-movflags",
        "+faststart",
        str(output),
    ]


def build_model_commands(manifest: dict[str, Any], root: Path) -> list[list[str]]:
    """Build existing CLI commands that create the canonical FP16 ONNX files."""
    model = manifest["model"]
    weights = repo_path(root, model["weights_path"])
    onnx_dir = repo_path(root, model["onnx_dir"])
    commands = [
        [
            "export-onnx",
            "--model_path",
            str(weights),
            "--output_dir",
            str(onnx_dir),
            "--quiet",
        ]
    ]
    for variant in model["variants"]:
        commands.append(
            [
                "prepare-onnx",
                str(repo_path(root, variant["fp32_path"])),
                "--output_dir",
                str(onnx_dir),
                "--precision",
                "fp16",
                "--quiet",
            ]
        )
    return commands


def run_command(command: list[str]) -> None:
    """Run one preparation command with readable diagnostics."""
    print(f"Running: {shlex.join(command)}")
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkloadError(f"Command failed: {shlex.join(command)}") from exc


def probe_video(path: Path) -> dict[str, Any]:
    """Return ffprobe stream and format metadata."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,"
            "avg_frame_rate,nb_frames,nb_read_frames,sample_aspect_ratio,color_range,"
            "color_space,color_transfer,color_primaries:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"Cannot probe video {path}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"Unexpected ffprobe output for {path}")
    return value


def validate_video_probe(
    probe: dict[str, Any],
    *,
    variant: dict[str, Any],
    clip: dict[str, Any],
) -> None:
    """Validate one prepared input against the canonical media contract."""
    streams = probe.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise WorkloadError("Prepared clip must contain exactly one stream")
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise WorkloadError("Prepared clip must contain one video stream")

    expected = {
        "codec_name": "h264",
        "width": variant["width"],
        "height": variant["height"],
        "pix_fmt": clip["encode"]["pixel_format"],
        "color_range": clip["encode"]["color_range"],
        "color_space": clip["encode"]["color_space"],
        "color_transfer": clip["encode"]["color_transfer"],
        "color_primaries": clip["encode"]["color_primaries"],
        "sample_aspect_ratio": "1:1",
    }
    for key, expected_value in expected.items():
        if stream.get(key) != expected_value:
            raise WorkloadError(
                f"Prepared clip metadata mismatch for {key}: "
                f"expected {expected_value}, got {stream.get(key)}"
            )

    expected_fps = Fraction(clip["fps"])
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            actual_fps = Fraction(stream.get(key, "0/1"))
        except (ValueError, ZeroDivisionError) as exc:
            raise WorkloadError(f"Prepared clip has invalid {key}") from exc
        if actual_fps != expected_fps:
            raise WorkloadError(f"Prepared clip {key} must be {expected_fps}, got {actual_fps}")

    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if str(frame_value) != str(clip["frames"]):
        raise WorkloadError(
            f"Prepared clip must contain {clip['frames']} frames, got {frame_value}"
        )
    try:
        duration = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkloadError("Prepared clip duration is unavailable") from exc
    expected_duration = clip["frames"] / float(expected_fps)
    if abs(duration - expected_duration) > 1 / float(expected_fps):
        raise WorkloadError(
            f"Prepared clip duration must be {expected_duration:.6f}, got {duration:.6f}"
        )


def verify_onnx(path: Path, *, width: int, height: int) -> dict[str, Any]:
    """Validate static mixed-FP16 ONNX with FP32 graph I/O."""
    if not path.is_file():
        raise WorkloadError(f"Prepared ONNX is missing: {path}")
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
    except (ImportError, OSError) as exc:
        raise WorkloadError(f"Cannot load ONNX {path}: {exc}") from exc

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise WorkloadError(f"ONNX must have one input and one output: {path}")

    def tensor_shape(value: Any) -> list[int]:
        return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]

    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]
    expected_input = [1, 3, height, width]
    expected_output = [1, 3, height * 2, width * 2]
    if tensor_shape(input_tensor) != expected_input:
        raise WorkloadError(f"Unexpected ONNX input shape in {path}")
    if tensor_shape(output_tensor) != expected_output:
        raise WorkloadError(f"Unexpected ONNX output shape in {path}")
    if input_tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise WorkloadError(f"ONNX input must remain FP32: {path}")
    if output_tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise WorkloadError(f"ONNX output must remain FP32: {path}")
    has_fp16_weights = any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in model.graph.initializer
    )
    if not has_fp16_weights:
        raise WorkloadError(f"ONNX does not contain FP16 initializers: {path}")

    return {
        "input_shape": expected_input,
        "output_shape": expected_output,
        "io_precision": "fp32",
        "internal_precision": "fp16",
    }


def prepare_clips(manifest: dict[str, Any], root: Path, *, force: bool) -> None:
    """Create deterministic compressed inputs from the lossless source."""
    clip = manifest["clip"]
    source = repo_path(root, clip["source_path"])
    for variant in clip["variants"]:
        output = repo_path(root, variant["path"])
        if output.exists() and not force:
            try:
                validate_video_probe(probe_video(output), variant=variant, clip=clip)
            except WorkloadError as exc:
                raise WorkloadError(f"Invalid existing clip {output}; rerun with --force") from exc
            print(f"Using verified clip: {output}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        run_command(build_ffmpeg_command(manifest, source, variant, output))
        validate_video_probe(probe_video(output), variant=variant, clip=clip)


def prepare_model(manifest: dict[str, Any], root: Path, *, force: bool) -> None:
    """Export static FP32 ONNX variants and convert them to mixed FP16."""
    model = manifest["model"]
    variants = model["variants"]
    if not force:
        try:
            for variant in variants:
                verify_onnx(
                    repo_path(root, variant["fp16_path"]),
                    width=variant["input_width"],
                    height=variant["input_height"],
                )
        except WorkloadError:
            pass
        else:
            print("Using verified mixed-FP16 ONNX variants")
            return

    repo_path(root, model["onnx_dir"]).mkdir(parents=True, exist_ok=True)
    for command in build_model_commands(manifest, root):
        run_command(command)


def verify_assets(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """Verify all workload assets and return sanitized lock data."""
    model = manifest["model"]
    clip = manifest["clip"]
    weight_path = repo_path(root, model["weights_path"])
    source_path = repo_path(root, clip["source_path"])
    weight_hash = verify_source_file(weight_path, model["source"])
    source_hash = verify_source_file(source_path, clip["source"])

    assets: list[dict[str, Any]] = []
    for variant in clip["variants"]:
        path = repo_path(root, variant["path"])
        validate_video_probe(probe_video(path), variant=variant, clip=clip)
        assets.append(
            {
                "kind": "input_video",
                "variant": variant["name"],
                "path": variant["path"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "width": variant["width"],
                "height": variant["height"],
                "frames": clip["frames"],
                "fps": clip["fps"],
            }
        )

    for variant in model["variants"]:
        path = repo_path(root, variant["fp16_path"])
        onnx_contract = verify_onnx(
            path,
            width=variant["input_width"],
            height=variant["input_height"],
        )
        assets.append(
            {
                "kind": "onnx",
                "variant": variant["name"],
                "path": variant["fp16_path"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                **onnx_contract,
            }
        )

    return {
        "schema_version": 1,
        "workload_id": manifest["id"],
        "sources": {
            "model": {
                "sha256": weight_hash,
                "size_bytes": weight_path.stat().st_size,
                "attribution": model["source"]["attribution"],
            },
            "clip": {
                "sha256": source_hash,
                "size_bytes": source_path.stat().st_size,
                "license": clip["source"].get("license"),
                "attribution": clip["source"]["attribution"],
            },
        },
        "assets": assets,
    }


def write_lock(path: Path, lock: dict[str, Any]) -> None:
    """Write the local generated-assets lock without host-specific paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(lock, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Asset lock written: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace invalid/generated assets during prepare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        root = args.root.resolve()
        if args.action == "prepare":
            model = manifest["model"]
            clip = manifest["clip"]
            download_source(
                model["source"],
                repo_path(root, model["weights_path"]),
                force=args.force,
            )
            download_source(
                clip["source"],
                repo_path(root, clip["source_path"]),
                force=args.force,
            )
            prepare_clips(manifest, root, force=args.force)
            prepare_model(manifest, root, force=args.force)
            lock = verify_assets(manifest, root)
            write_lock(repo_path(root, manifest["lock_path"]), lock)
        else:
            verify_assets(manifest, root)
            print(f"Workload verified: {manifest['id']}")
    except WorkloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
