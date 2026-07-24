"""Cached model preparation, engine build, inference, and reporting for the demo."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_media.demo import DemoError
from ai_media.demo.config import (
    DEMO_BITRATE_MBPS,
    DEMO_INPUT_HEIGHT,
    DEMO_INPUT_WIDTH,
    DEMO_OUTPUT_HEIGHT,
    DEMO_OUTPUT_WIDTH,
    MODEL_ATTRIBUTION,
    MODEL_LICENSE_URL,
    MODEL_NAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    MODEL_URL,
    DemoPaths,
    DemoVideoContract,
)
from ai_media.demo.media import (
    build_demo_input_command,
    validate_demo_video,
    write_demo_media_assets,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoError(f"command failed: {shlex.join(command)}") from exc


def verify_pinned_weights(path: Path) -> bool:
    """Return whether a cached model exactly matches the pinned source."""
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE_BYTES
        and _sha256_file(path) == MODEL_SHA256
    )


def download_weights(path: Path, *, force: bool) -> None:
    """Download the pinned model and verify size plus SHA256 before use."""
    if verify_pinned_weights(path):
        print(f"Using verified weights: {path}")
        return
    if path.exists() and not force:
        raise DemoError(
            f"cached weights are invalid: {path}. Rerun with DEMO_FORCE=1 to replace them."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "ai-media-enhancer-demo/1"},
    )
    print(f"Downloading pinned {MODEL_NAME} weights ({MODEL_SIZE_BYTES / 1e6:.1f} MB)...")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    except OSError as exc:
        raise DemoError(f"model download failed: {exc}") from exc

    if not verify_pinned_weights(partial):
        partial.unlink(missing_ok=True)
        raise DemoError("downloaded model failed size/SHA256 verification")
    os.replace(partial, path)
    print(f"Verified weights: {path}")


def _valid_onnx(path: Path, *, fp16: bool) -> bool:
    if not path.is_file():
        return False
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
    except (ImportError, OSError):
        return False
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return False

    def shape(value: Any) -> list[int]:
        return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]

    if shape(model.graph.input[0]) != [1, 3, DEMO_INPUT_HEIGHT, DEMO_INPUT_WIDTH]:
        return False
    if shape(model.graph.output[0]) != [1, 3, DEMO_OUTPUT_HEIGHT, DEMO_OUTPUT_WIDTH]:
        return False
    if not fp16:
        return True
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        return False
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        return False
    return any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in model.graph.initializer
    )


def _valid_engine_cache(paths: DemoPaths) -> bool:
    if not paths.engine.is_file() or not paths.engine_manifest.is_file():
        return False
    try:
        import tensorrt as trt

        manifest = json.loads(paths.engine_manifest.read_text(encoding="utf-8"))
    except (ImportError, OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("engine_sha256") == _sha256_file(paths.engine)
        and manifest.get("model_sha256") == _sha256_file(paths.fp16_onnx)
        and manifest.get("tensorrt_version") == trt.__version__
        and manifest.get("input", {}).get("shape")
        == [1, 3, DEMO_INPUT_HEIGHT, DEMO_INPUT_WIDTH]
        and manifest.get("output", {}).get("shape")
        == [1, 3, DEMO_OUTPUT_HEIGHT, DEMO_OUTPUT_WIDTH]
    )


def _ensure_cached(
    path: Path,
    validator: Callable[[], bool],
    command: list[str],
    *,
    force: bool,
) -> None:
    if not force and validator():
        print(f"Using validated cache: {path}")
        return
    path.unlink(missing_ok=True)
    _run(command)
    if not validator():
        raise DemoError(f"generated artifact failed validation: {path}")


def _prepare_input(paths: DemoPaths, *, force: bool) -> dict[str, Any]:
    contract = DemoVideoContract(DEMO_INPUT_WIDTH, DEMO_INPUT_HEIGHT)
    if not force and paths.input_video.is_file():
        try:
            observed = validate_demo_video(paths.input_video, contract)
        except DemoError:
            pass
        else:
            print(f"Using validated input: {paths.input_video}")
            return observed

    write_demo_media_assets(paths)
    paths.input_video.parent.mkdir(parents=True, exist_ok=True)
    paths.input_video.unlink(missing_ok=True)
    _run(build_demo_input_command(paths))
    return validate_demo_video(paths.input_video, contract)


def _prepare_model(paths: DemoPaths, *, force: bool) -> None:
    paths.onnx_dir.mkdir(parents=True, exist_ok=True)
    _ensure_cached(
        paths.fp32_onnx,
        lambda: _valid_onnx(paths.fp32_onnx, fp16=False),
        [
            "export-onnx",
            "--model_path",
            str(paths.weights),
            "--output_dir",
            str(paths.onnx_dir),
            "--name",
            "realesrgan_x2plus",
            "--size",
            "1280x720",
        ],
        force=force,
    )
    _ensure_cached(
        paths.fp16_onnx,
        lambda: _valid_onnx(paths.fp16_onnx, fp16=True),
        [
            "prepare-onnx",
            str(paths.fp32_onnx),
            "--output_dir",
            str(paths.onnx_dir),
            "--precision",
            "fp16",
        ],
        force=force,
    )


def _build_engine(paths: DemoPaths, *, force: bool) -> bool:
    reused = not force and _valid_engine_cache(paths)
    if reused:
        print(f"Using validated engine cache: {paths.engine}")
        return True

    paths.engine.parent.mkdir(parents=True, exist_ok=True)
    paths.timing_cache.parent.mkdir(parents=True, exist_ok=True)
    paths.engine.unlink(missing_ok=True)
    paths.engine_manifest.unlink(missing_ok=True)
    if force:
        paths.timing_cache.unlink(missing_ok=True)
    _run(
        [
            "build-engine",
            str(paths.fp16_onnx),
            "--output",
            str(paths.engine),
            "--timing-cache",
            str(paths.timing_cache),
        ]
    )
    if not _valid_engine_cache(paths):
        raise DemoError("TensorRT engine or sidecar failed cache validation")
    return False


def upscale_command(paths: DemoPaths, gpu_id: int) -> list[str]:
    """Build the canonical demo inference command."""
    return [
        "upscale",
        "--backend",
        "nvcodec",
        "--gpu-id",
        str(gpu_id),
        "--engine",
        str(paths.engine),
        "--input",
        str(paths.input_video),
        "--output",
        str(paths.output_video),
        "--bitrate-mbps",
        str(DEMO_BITRATE_MBPS),
        "--log-interval",
        "8",
    ]


def _run_upscale(paths: DemoPaths, *, gpu_id: int, engine_reused: bool) -> None:
    paths.output_video.parent.mkdir(parents=True, exist_ok=True)
    command = upscale_command(paths, gpu_id)
    try:
        _run(command)
    except DemoError:
        if not engine_reused:
            raise
        print("Cached engine failed at runtime; rebuilding it for the current GPU...")
        _build_engine(paths, force=True)
        _run(command)


def _write_report(
    paths: DemoPaths,
    *,
    input_observed: dict[str, Any],
    output_observed: dict[str, Any],
) -> None:
    def asset(path: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(paths.root).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    report = {
        "schema_version": 1,
        "status": "valid",
        "model": {
            "name": MODEL_NAME,
            "source_url": MODEL_URL,
            "sha256": MODEL_SHA256,
            "attribution": MODEL_ATTRIBUTION,
            "license": MODEL_LICENSE_URL,
        },
        "assets": {
            "weights": asset(paths.weights),
            "onnx": asset(paths.fp16_onnx),
            "engine": asset(paths.engine),
            "input": asset(paths.input_video),
            "output": asset(paths.output_video),
        },
        "input": input_observed,
        "output": output_observed,
    }
    paths.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_demo(root: Path, *, gpu_id: int, force: bool) -> DemoPaths:
    """Execute the complete cached model-to-video demonstration."""
    paths = DemoPaths.under(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL_ATTRIBUTION}")
    print(f"License: {MODEL_LICENSE_URL}")

    download_weights(paths.weights, force=force)
    input_observed = _prepare_input(paths, force=force)
    _prepare_model(paths, force=force)
    engine_reused = _build_engine(paths, force=force)
    _run_upscale(paths, gpu_id=gpu_id, engine_reused=engine_reused)
    output_observed = validate_demo_video(
        paths.output_video,
        DemoVideoContract(DEMO_OUTPUT_WIDTH, DEMO_OUTPUT_HEIGHT),
    )
    _write_report(
        paths,
        input_observed=input_observed,
        output_observed=output_observed,
    )
    return paths
