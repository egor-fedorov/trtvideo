"""Validate benchmark workload manifests and resolve their paths safely."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkloadError(RuntimeError):
    """Raised when workload preparation or validation fails."""


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


def find_clip_variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one canonical clip variant by name."""
    variants = manifest.get("clip", {}).get("variants", [])
    for variant in variants:
        if isinstance(variant, dict) and variant.get("name") == name:
            return variant
    available = ", ".join(str(item.get("name")) for item in variants)
    raise WorkloadError(f"Unknown variant {name!r}; available: {available}")


def find_model_variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one canonical model variant by name."""
    variants = manifest.get("model", {}).get("variants", [])
    for variant in variants:
        if isinstance(variant, dict) and variant.get("name") == name:
            return variant
    raise WorkloadError(f"Workload has no model variant {name!r}")


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


def _validate_temporal_sampling(clip: dict[str, Any]) -> None:
    sampling = clip.get("temporal_sampling")
    if sampling is None:
        return
    if not isinstance(sampling, dict):
        raise WorkloadError("Manifest field 'clip.temporal_sampling' must be an object")
    try:
        source_fps = Fraction(str(sampling.get("source_fps")))
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkloadError(
            "Manifest field 'clip.temporal_sampling.source_fps' must be a rational FPS"
        ) from exc
    if source_fps <= 0:
        raise WorkloadError("Manifest field 'clip.temporal_sampling.source_fps' must be positive")
    target_fps = Fraction(str(clip["fps"]))
    if source_fps < target_fps:
        raise WorkloadError(
            "Manifest field 'clip.temporal_sampling.source_fps' cannot be lower "
            "than clip.fps when using frame dropping"
        )
    if sampling.get("method") != "drop":
        raise WorkloadError("Manifest field 'clip.temporal_sampling.method' must be 'drop'")
    if sampling.get("round") != "near":
        raise WorkloadError("Manifest field 'clip.temporal_sampling.round' must be 'near'")


def _validate_model_space_quality(manifest: dict[str, Any], *, clip_frames: int) -> None:
    quality = _require_dict(manifest, "quality")
    model_space = _require_dict(quality, "model_space")
    frame_indices = model_space.get("frame_indices")
    if (
        not isinstance(frame_indices, list)
        or not frame_indices
        or not all(isinstance(value, int) for value in frame_indices)
    ):
        raise WorkloadError(
            "Manifest field 'quality.model_space.frame_indices' must be a non-empty integer array"
        )
    if frame_indices != sorted(set(frame_indices)):
        raise WorkloadError(
            "Manifest field 'quality.model_space.frame_indices' must be sorted and unique"
        )
    if frame_indices[0] < 0 or frame_indices[-1] >= clip_frames:
        raise WorkloadError(
            "Manifest model-space frame indices must stay inside the canonical clip"
        )
    if model_space.get("contract_version") != 3:
        raise WorkloadError("Manifest model-space contract_version must be 3")

    inference = _require_dict(model_space, "inference")
    if inference.get("canonical_input") != "trtvideo-production-rgb-f32":
        raise WorkloadError("Manifest model-space canonical input contract is invalid")
    if inference.get("input_acceptance") != "exact-float32":
        raise WorkloadError("Manifest model-space input acceptance contract is invalid")
    output_thresholds = _require_dict(inference, "output_thresholds")
    for name in ("p99_abs", "rmse", "min_psnr_db"):
        value = output_thresholds.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise WorkloadError(
                "Manifest field "
                f"'quality.model_space.inference.output_thresholds.{name}' "
                "must be positive"
            )

    preprocessing = _require_dict(model_space, "preprocessing")
    if preprocessing.get("acceptance") != "diagnostic-only":
        raise WorkloadError("Manifest model-space preprocessing must be diagnostic-only")


def _validate_product_output_quality(
    manifest: dict[str, Any],
    *,
    clip_frames: int,
) -> None:
    quality = _require_dict(manifest, "quality")
    product_output = _require_dict(quality, "product_output")
    frame_indices = product_output.get("frame_indices")
    if (
        not isinstance(frame_indices, list)
        or not frame_indices
        or not all(isinstance(value, int) for value in frame_indices)
        or frame_indices != sorted(set(frame_indices))
    ):
        raise WorkloadError(
            "Manifest field 'quality.product_output.frame_indices' "
            "must be a sorted unique integer array"
        )
    if frame_indices[0] < 0 or frame_indices[-1] >= clip_frames:
        raise WorkloadError(
            "Manifest product-output frame indices must stay inside the canonical clip"
        )

    thresholds = _require_dict(product_output, "thresholds")
    psnr_min_db = thresholds.get("psnr_min_db")
    ssim_min = thresholds.get("ssim_min")
    if (
        not isinstance(psnr_min_db, (int, float))
        or isinstance(psnr_min_db, bool)
        or psnr_min_db <= 0
    ):
        raise WorkloadError(
            "Manifest field 'quality.product_output.thresholds.psnr_min_db' must be positive"
        )
    if (
        not isinstance(ssim_min, (int, float))
        or isinstance(ssim_min, bool)
        or not 0 < ssim_min <= 1
    ):
        raise WorkloadError(
            "Manifest field 'quality.product_output.thresholds.ssim_min' must be in (0, 1]"
        )

    crops = _require_list(product_output, "crops")
    names: set[str] = set()
    for crop in crops:
        name = crop.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise WorkloadError(
                "Manifest product-output crop names must be unique non-empty strings"
            )
        names.add(name)
        for field in ("x", "y", "width", "height"):
            value = crop.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
                or (field in {"width", "height"} and value <= 0)
                or value > 1
            ):
                raise WorkloadError(f"Manifest product-output crop '{name}' has invalid {field}")
        if crop["x"] + crop["width"] > 1 or crop["y"] + crop["height"] > 1:
            raise WorkloadError(f"Manifest product-output crop '{name}' exceeds the output frame")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the canonical workload schema and path boundaries."""
    if manifest.get("schema_version") != 1:
        raise WorkloadError("Unsupported workload schema_version")
    if not isinstance(manifest.get("id"), str) or not manifest["id"]:
        raise WorkloadError("Manifest field 'id' is required")
    _validate_relative_path(manifest.get("lock_path"), "lock_path")
    benchmark = _require_dict(manifest, "benchmark")
    required_benchmark_fields = {
        "contract_version",
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
        "contract_version",
        "warmup_frames",
        "measured_frames",
        "initial_runs",
        "nvml_sample_interval_ms",
    ):
        if (
            not isinstance(benchmark.get(field), int)
            or isinstance(benchmark[field], bool)
            or benchmark[field] <= 0
        ):
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
    if not isinstance(model.get("export_name"), str) or not model["export_name"]:
        raise WorkloadError("Manifest field 'model.export_name' is required")
    export_conformance = _require_dict(model, "export_conformance")
    if export_conformance.get("contract_version") != 1:
        raise WorkloadError("Manifest model export-conformance contract_version must be 1")
    _validate_relative_path(
        export_conformance.get("report_path"),
        "model.export_conformance.report_path",
    )
    _validate_relative_path(model.get("weights_path"), "model.weights_path")
    _validate_relative_path(model.get("onnx_dir"), "model.onnx_dir")
    model_variants = _require_list(model, "variants")

    clip = _require_dict(manifest, "clip")
    _validate_source(_require_dict(clip, "source"), "clip.source")
    _validate_relative_path(clip.get("source_path"), "clip.source_path")
    if clip.get("frames") != 1000:
        raise WorkloadError("Canonical workload must contain exactly 1000 frames")
    _validate_model_space_quality(manifest, clip_frames=clip["frames"])
    _validate_product_output_quality(manifest, clip_frames=clip["frames"])
    try:
        fps = Fraction(str(clip.get("fps")))
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkloadError("Manifest field 'clip.fps' must be a rational FPS") from exc
    if fps <= 0:
        raise WorkloadError("Manifest field 'clip.fps' must be positive")
    _validate_temporal_sampling(clip)
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
    if not isinstance(encode.get("b_frames"), int) or encode["b_frames"] < 0:
        raise WorkloadError("Manifest field 'clip.encode.b_frames' must be non-negative")
    ffprobe_has_b_frames = encode.get("ffprobe_has_b_frames", encode["b_frames"])
    if not isinstance(ffprobe_has_b_frames, int) or ffprobe_has_b_frames < 0:
        raise WorkloadError(
            "Manifest field 'clip.encode.ffprobe_has_b_frames' must be non-negative"
        )
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
            raise WorkloadError(f"Clip variant '{name}' has invalid benchmark_output.bitrate_mbps")

    if model_names != clip_names:
        raise WorkloadError("Model and clip variants must use the same names")


def repo_path(root: Path, relative_path: str) -> Path:
    """Resolve a validated repository-relative path."""
    root = root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise WorkloadError(f"Path escapes repository root: {relative_path}")
    return resolved
