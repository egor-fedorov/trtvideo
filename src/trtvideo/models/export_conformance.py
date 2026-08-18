"""Semantic conformance contract for Spandrel-to-ONNX model exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

EXPORT_CONFORMANCE_DOCUMENT_TYPE = "model-export-conformance"
EXPORT_CONFORMANCE_SCHEMA_VERSION = 2
EXPORT_CONTRACT_METADATA_KEY = "trtvideo.export_contract"
EXPORT_CONTRACT_METADATA_VALUE = "spandrel-image-upscale-v2"
EXPORT_SCALE_METADATA_KEY = "trtvideo.scale"
EXPORT_PROBE_VERSION = "rgb-coordinate-pattern-v1"
EXPORT_PROBE_HEIGHT = 16
EXPORT_PROBE_WIDTH = 16
EXPORT_PROBE_SHA256 = "77565293a44ee854a6c2490773756fe6a816bebd57b59dfc7b71dde6a4b10ab8"
EXPORT_MAX_ABS_THRESHOLD = 1e-4
EXPORT_RMSE_THRESHOLD = 1e-5
EXPORT_MIN_PSNR_DB = 80.0

_EXPORT_TOOL_PACKAGES = ("torch", "onnx", "onnxruntime", "onnxscript", "spandrel")


class ExportConformanceError(RuntimeError):
    """Raised when an exported ONNX graph is not equivalent to its source model."""


def infer_upscale_scale(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> int:
    """Return the uniform integer scale represented by two RGB NCHW shapes."""
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ExportConformanceError(
            f"Expected rank-4 NCHW tensors, got {list(input_shape)} -> {list(output_shape)}"
        )
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in (*input_shape, *output_shape)
    ):
        raise ExportConformanceError(
            f"Expected static positive tensor shapes, got {list(input_shape)} -> "
            f"{list(output_shape)}"
        )

    in_n, in_c, in_h, in_w = input_shape
    out_n, out_c, out_h, out_w = output_shape
    if (in_n, in_c, out_n, out_c) != (1, 3, 1, 3):
        raise ExportConformanceError(
            "Expected batch-1 RGB input/output tensors, got "
            f"{list(input_shape)} -> {list(output_shape)}"
        )
    if out_h % in_h != 0 or out_w % in_w != 0:
        raise ExportConformanceError(
            "Output shape must be an integer spatial scale of the input, got "
            f"{list(input_shape)} -> {list(output_shape)}"
        )
    scale_h = out_h // in_h
    scale_w = out_w // in_w
    if scale_h != scale_w:
        raise ExportConformanceError(
            f"Output scale must be uniform, got {scale_w}x horizontally and {scale_h}x vertically"
        )
    return scale_h


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_tool_versions() -> dict[str, str]:
    """Return versions that affect Spandrel-to-ONNX graph generation."""
    return {package: version(package) for package in _EXPORT_TOOL_PACKAGES}


def conformance_report_path(output_dir: Path, model_name: str) -> Path:
    """Return the default evidence path for one exported model."""
    return output_dir / f"{model_name}.export-conformance.json"


def deterministic_probe(torch: Any) -> Any:
    """Build a small RGB tensor whose channels and spatial offsets are distinct."""
    y = torch.arange(EXPORT_PROBE_HEIGHT, dtype=torch.int64).reshape(
        EXPORT_PROBE_HEIGHT,
        1,
    )
    x = torch.arange(EXPORT_PROBE_WIDTH, dtype=torch.int64).reshape(
        1,
        EXPORT_PROBE_WIDTH,
    )
    channels = (
        (x * 13 + y * 7) % 251,
        (x * 5 + y * 17 + 37) % 251,
        (x * 19 + y * 3 + 83) % 251,
    )
    return torch.stack(channels).to(dtype=torch.float32).div_(250.0).unsqueeze(0)


def compare_outputs(reference: Any, candidate: Any, torch: Any) -> dict[str, float]:
    """Validate one ONNX output against its source-model FP32 reference."""
    reference = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ExportConformanceError(
            "Export probe shape mismatch: "
            f"PyTorch {tuple(reference.shape)}, ONNX {tuple(candidate.shape)}"
        )
    if not bool(torch.isfinite(reference).all()):
        raise ExportConformanceError("PyTorch export probe produced non-finite values")
    if not bool(torch.isfinite(candidate).all()):
        raise ExportConformanceError("ONNX export probe produced non-finite values")

    difference = candidate - reference
    max_abs = float(difference.abs().max().item())
    mean_abs = float(difference.abs().mean().item())
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_l2 = float(torch.linalg.vector_norm(reference).item())
    difference_l2 = float(torch.linalg.vector_norm(difference).item())
    relative_l2 = difference_l2 / reference_l2 if reference_l2 else difference_l2
    psnr_db = 200.0 if rmse == 0 else 20.0 * math.log10(1.0 / rmse)

    errors = []
    if max_abs > EXPORT_MAX_ABS_THRESHOLD:
        errors.append(
            f"max_abs must be <= {EXPORT_MAX_ABS_THRESHOLD:g}, got {max_abs:.9g}"
        )
    if rmse > EXPORT_RMSE_THRESHOLD:
        errors.append(f"RMSE must be <= {EXPORT_RMSE_THRESHOLD:g}, got {rmse:.9g}")
    if psnr_db < EXPORT_MIN_PSNR_DB:
        errors.append(f"PSNR must be >= {EXPORT_MIN_PSNR_DB:g} dB, got {psnr_db:.6f} dB")
    if errors:
        raise ExportConformanceError("Exported ONNX differs from PyTorch: " + "; ".join(errors))

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "psnr_db": psnr_db,
    }


def build_conformance_report(
    *,
    model_name: str,
    weights_path: Path,
    probe: Any,
    output_shape: list[int],
    metrics: dict[str, float],
    exported_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Build immutable evidence for one source-model export invocation."""
    scale = infer_upscale_scale(list(probe.shape), output_shape)
    probe_sha256 = hashlib.sha256(
        probe.detach().to(device="cpu").contiguous().numpy().tobytes()
    ).hexdigest()
    if probe_sha256 != EXPORT_PROBE_SHA256:
        raise ExportConformanceError("Generated export-conformance probe hash does not match")
    exports = []
    for path in sorted(exported_paths):
        exports.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "document_type": EXPORT_CONFORMANCE_DOCUMENT_TYPE,
        "schema_version": EXPORT_CONFORMANCE_SCHEMA_VERSION,
        "status": "valid",
        "export_contract": EXPORT_CONTRACT_METADATA_VALUE,
        "model": {
            "name": model_name,
            "scale": scale,
            "source_sha256": sha256_file(weights_path),
            "source_size_bytes": weights_path.stat().st_size,
        },
        "probe": {
            "version": EXPORT_PROBE_VERSION,
            "input_shape": list(probe.shape),
            "input_sha256": probe_sha256,
            "output_shape": output_shape,
        },
        "comparison": {
            "reference": "pytorch-fp32",
            "candidate": "onnxruntime-cpu-fp32",
            "thresholds": {
                "max_abs": EXPORT_MAX_ABS_THRESHOLD,
                "rmse": EXPORT_RMSE_THRESHOLD,
                "min_psnr_db": EXPORT_MIN_PSNR_DB,
            },
            "metrics": metrics,
        },
        "tools": export_tool_versions(),
        "exports": exports,
    }


def write_conformance_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write export-conformance evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_conformance_report(
    report: dict[str, Any],
    *,
    model_name: str,
    source_sha256: str,
    source_size_bytes: int,
    exported_files: dict[str, tuple[str, int]],
    tool_versions: dict[str, str] | None = None,
    expected_scale: int | None = None,
) -> dict[str, Any]:
    """Validate cached evidence and return its comparison summary."""
    expected = {
        "document_type": EXPORT_CONFORMANCE_DOCUMENT_TYPE,
        "schema_version": EXPORT_CONFORMANCE_SCHEMA_VERSION,
        "status": "valid",
        "export_contract": EXPORT_CONTRACT_METADATA_VALUE,
    }
    for field, expected_value in expected.items():
        if report.get(field) != expected_value:
            raise ExportConformanceError(
                f"Export-conformance field {field!r} must be {expected_value!r}"
            )

    model = report.get("model")
    if not isinstance(model, dict) or any(
        model.get(field) != value
        for field, value in {
            "name": model_name,
            "source_sha256": source_sha256,
            "source_size_bytes": source_size_bytes,
        }.items()
    ):
        raise ExportConformanceError("Export-conformance source model identity does not match")
    scale = model.get("scale")
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ExportConformanceError("Export-conformance model scale is invalid")
    if expected_scale is not None and scale != expected_scale:
        raise ExportConformanceError(
            f"Export-conformance model scale must be {expected_scale}x, got {scale}x"
        )

    probe = report.get("probe")
    expected_input_shape = [1, 3, EXPORT_PROBE_HEIGHT, EXPORT_PROBE_WIDTH]
    if not isinstance(probe, dict) or probe.get("version") != EXPORT_PROBE_VERSION:
        raise ExportConformanceError("Export-conformance probe version does not match")
    if probe.get("input_shape") != expected_input_shape:
        raise ExportConformanceError("Export-conformance probe input shape does not match")
    output_shape = probe.get("output_shape")
    if not isinstance(output_shape, list):
        raise ExportConformanceError("Export-conformance probe output shape is invalid")
    observed_scale = infer_upscale_scale(expected_input_shape, output_shape)
    if observed_scale != scale:
        raise ExportConformanceError(
            "Export-conformance probe output shape does not match its model scale"
        )
    if probe.get("input_sha256") != EXPORT_PROBE_SHA256:
        raise ExportConformanceError("Export-conformance probe input hash does not match")

    comparison = report.get("comparison")
    expected_thresholds = {
        "max_abs": EXPORT_MAX_ABS_THRESHOLD,
        "rmse": EXPORT_RMSE_THRESHOLD,
        "min_psnr_db": EXPORT_MIN_PSNR_DB,
    }
    if not isinstance(comparison, dict):
        raise ExportConformanceError("Export-conformance comparison is missing")
    if comparison.get("reference") != "pytorch-fp32":
        raise ExportConformanceError("Export-conformance reference is invalid")
    if comparison.get("candidate") != "onnxruntime-cpu-fp32":
        raise ExportConformanceError("Export-conformance candidate is invalid")
    if comparison.get("thresholds") != expected_thresholds:
        raise ExportConformanceError("Export-conformance thresholds do not match")
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        raise ExportConformanceError("Export-conformance metrics are missing")
    for name in ("max_abs", "mean_abs", "rmse", "relative_l2", "psnr_db"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ExportConformanceError(f"Export-conformance metric {name!r} is invalid")
    if any(metrics[name] < 0 for name in ("max_abs", "mean_abs", "rmse", "relative_l2")):
        raise ExportConformanceError("Export-conformance error metrics cannot be negative")
    if metrics["max_abs"] > EXPORT_MAX_ABS_THRESHOLD:
        raise ExportConformanceError("Export-conformance max_abs exceeds its threshold")
    if metrics["rmse"] > EXPORT_RMSE_THRESHOLD:
        raise ExportConformanceError("Export-conformance RMSE exceeds its threshold")
    if metrics["psnr_db"] < EXPORT_MIN_PSNR_DB:
        raise ExportConformanceError("Export-conformance PSNR is below its threshold")

    expected_tools = tool_versions if tool_versions is not None else export_tool_versions()
    if report.get("tools") != expected_tools:
        raise ExportConformanceError("Export-conformance tool versions do not match")

    exports = report.get("exports")
    if not isinstance(exports, list):
        raise ExportConformanceError("Export-conformance exports are missing")
    observed_exports: dict[str, tuple[str, int]] = {}
    for item in exports:
        if not isinstance(item, dict):
            raise ExportConformanceError("Export-conformance export entry is invalid")
        path = item.get("path")
        sha256 = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if (
            not isinstance(path, str)
            or not isinstance(sha256, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or path in observed_exports
        ):
            raise ExportConformanceError("Export-conformance export entry is invalid")
        observed_exports[path] = (sha256, size_bytes)
    if observed_exports != exported_files:
        raise ExportConformanceError("Export-conformance ONNX identities do not match")
    return comparison
