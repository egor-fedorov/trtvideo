"""TensorRT engine contracts shared by runners, quality gates, and diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchmarks.scripts.runtime.environment import sha256_file
from benchmarks.scripts.workloads.manifest import find_clip_variant, find_model_variant


class EngineContractError(RuntimeError):
    """Raised when an engine or its build metadata violates the benchmark contract."""


def load_engine_contract(engine: Path) -> tuple[dict[str, Any], Path]:
    """Load and verify the sidecar emitted by build-engine."""
    sidecar_path = Path(f"{engine}.json")
    if not sidecar_path.is_file():
        raise EngineContractError(f"Engine sidecar not found: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineContractError(f"Cannot read JSON {sidecar_path}: {exc}") from exc
    if not isinstance(sidecar, dict):
        raise EngineContractError(f"Expected a JSON object in {sidecar_path}")

    expected_hash = sidecar.get("engine_sha256")
    actual_hash = sha256_file(engine)
    if expected_hash != actual_hash:
        raise EngineContractError(
            f"Engine SHA256 does not match sidecar: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    for tensor_name in ("input", "output"):
        shape = sidecar.get(tensor_name, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 4 or not all(
            isinstance(value, int) and value > 0 for value in shape
        ):
            raise EngineContractError(
                f"Engine sidecar has invalid static {tensor_name} shape"
            )
    return sidecar, sidecar_path


def validate_static_engine_contract(
    sidecar: dict[str, Any],
    manifest: dict[str, Any],
    variant_name: str,
    onnx_path: Path,
) -> None:
    """Verify the tensor contract required by comparative benchmarks."""
    model_variant = find_model_variant(manifest, variant_name)
    clip_variant = find_clip_variant(manifest, variant_name)
    output = clip_variant["benchmark_output"]
    expected_input = [
        1,
        3,
        model_variant["input_height"],
        model_variant["input_width"],
    ]
    expected_output = [1, 3, output["height"], output["width"]]
    if sidecar.get("input", {}).get("shape") != expected_input:
        raise EngineContractError("Engine input shape does not match workload")
    if sidecar.get("output", {}).get("shape") != expected_output:
        raise EngineContractError("Engine output shape does not match workload")
    if sidecar.get("io_precision") != "fp32":
        raise EngineContractError("Benchmark engine must keep FP32 input/output bindings")
    if sidecar.get("input_profile") is not None:
        raise EngineContractError("Benchmark engine must use a static full-frame shape")
    if not onnx_path.is_file():
        raise EngineContractError(f"Canonical ONNX not found: {onnx_path}")
    if sidecar.get("model_sha256") != sha256_file(onnx_path):
        raise EngineContractError("Engine was not built from the canonical ONNX")


def validate_vsgan_engine_contract(
    sidecar: dict[str, Any],
    manifest: dict[str, Any],
    variant_name: str,
    onnx_path: Path,
    expected_base_image: str,
    expected_ffmpeg_package: str,
) -> None:
    """Verify the pinned VSGAN build and runtime provenance."""
    validate_static_engine_contract(sidecar, manifest, variant_name, onnx_path)
    if "stronglyTyped" not in sidecar.get("builder_flags", []):
        raise EngineContractError("VSGAN engine must be strongly typed")
    version = str(sidecar.get("tensorrt_version", "")).replace(".", "")
    if not version.startswith("1016"):
        raise EngineContractError("Pinned VSGAN engine must be built by TensorRT 10.16")

    builder_base_image = sidecar.get("builder_base_image")
    runtime_base_image = os.environ.get("TRTVIDEO_BASE_IMAGE", "unknown")
    if runtime_base_image != expected_base_image:
        raise EngineContractError(
            "VSGAN runtime does not match the pinned implementation "
            f"({runtime_base_image!r} != {expected_base_image!r}); rebuild the image"
        )
    runtime_ffmpeg_package = os.environ.get(
        "TRTVIDEO_VSGAN_FFMPEG_PACKAGE", "unknown"
    )
    if runtime_ffmpeg_package != expected_ffmpeg_package:
        raise EngineContractError(
            "VSGAN FFmpeg does not match the pinned implementation "
            f"({runtime_ffmpeg_package!r} != {expected_ffmpeg_package!r}); "
            "rebuild the image"
        )
    if builder_base_image != runtime_base_image:
        raise EngineContractError(
            "VSGAN engine was built with a different base image "
            f"({builder_base_image!r} != {runtime_base_image!r}); rebuild the engine"
        )
