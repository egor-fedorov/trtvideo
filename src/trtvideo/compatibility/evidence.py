"""Sanitized model and command evidence for compatibility reports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from trtvideo.models.export_conformance import (
    EXPORT_CONFORMANCE_DOCUMENT_TYPE,
    EXPORT_CONFORMANCE_SCHEMA_VERSION,
    EXPORT_CONTRACT_METADATA_VALUE,
    EXPORT_MAX_ABS_THRESHOLD,
    EXPORT_MIN_PSNR_DB,
    EXPORT_PROBE_HEIGHT,
    EXPORT_PROBE_SHA256,
    EXPORT_PROBE_VERSION,
    EXPORT_PROBE_WIDTH,
    EXPORT_RMSE_THRESHOLD,
    ExportConformanceError,
    infer_upscale_scale,
)
from trtvideo.models.manifest import ModelSpec, TensorDType, make_upscale_model_spec

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_COMMAND_PATTERNS = (
    (re.compile(r"/(?:home|Users)/[^/\s]+"), "absolute user-home path"),
    (re.compile(r"/root(?:/|\b)"), "root home path"),
    (re.compile(r"\bGPU-[0-9a-fA-F-]{16,}\b"), "GPU UUID"),
    (re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]+\b"), "GitHub token"),
    (
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
        "URL credential",
    ),
    (
        re.compile(r"(?i)\b(?:password|passwd|token|secret|api[_-]?key)\s*=\s*[^\s]+"),
        "inline credential",
    ),
    (re.compile(r"(?i)\bauthorization:\s*(?:bearer|basic)\s+\S+"), "authorization header"),
    (re.compile(r"(?m)^\s*ssh\s+\S+"), "SSH host identity"),
)
_EXPORT_TOOL_PACKAGES = ("torch", "onnx", "onnxruntime", "onnxscript", "spandrel")


class CompatibilityEvidenceError(ValueError):
    """Raised when supplied evidence is missing, inconsistent, or unsafe."""


def sha256_file(path: Path) -> str:
    """Hash one evidence artifact without exposing its path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Return a path-free immutable file identity."""
    try:
        if not path.is_file():
            raise CompatibilityEvidenceError(f"Evidence file does not exist: {path.name}")
        return {
            "name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    except CompatibilityEvidenceError:
        raise
    except OSError as exc:
        raise CompatibilityEvidenceError(f"Cannot read evidence file: {path.name}") from exc


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON object with an evidence-oriented error."""
    if not path.is_file():
        raise CompatibilityEvidenceError(f"{label} does not exist: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompatibilityEvidenceError(f"{label} is not valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise CompatibilityEvidenceError(f"{label} must contain a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CompatibilityEvidenceError(f"{label} must be a lowercase SHA256")
    return value


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise CompatibilityEvidenceError(f"{label} must be a static rank-4 shape")
    return tuple(value)


def _tensor_dtype(value: Any, label: str) -> TensorDType:
    normalized = str(value).lower()
    if normalized in {"datatype.float", "float", "float32", "fp32"}:
        return "fp32"
    if normalized in {"datatype.half", "half", "float16", "fp16"}:
        return "fp16"
    raise CompatibilityEvidenceError(f"{label} has unsupported dtype {value!r}")


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CompatibilityEvidenceError(f"{label} must be text")
    return public_value(value, label)


def _artifact_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompatibilityEvidenceError(f"{label} is missing")
    return public_value(re.split(r"[/\\]", value.strip())[-1], label)


def engine_evidence(engine_path: Path) -> tuple[dict[str, Any], ModelSpec]:
    """Validate an engine sidecar and return a sanitized summary and model spec."""
    identity = file_identity(engine_path)
    sidecar_path = Path(f"{engine_path}.json")
    sidecar = load_json_object(sidecar_path, "Engine sidecar")
    if sidecar.get("schema_version") != 1:
        raise CompatibilityEvidenceError("Engine sidecar schema_version must be 1")
    if sidecar.get("engine_sha256") != identity["sha256"]:
        raise CompatibilityEvidenceError("Engine hash does not match its sidecar")

    model_sha256 = _require_sha256(sidecar.get("model_sha256"), "Engine model_sha256")
    input_tensor = sidecar.get("input")
    output_tensor = sidecar.get("output")
    if not isinstance(input_tensor, dict) or not isinstance(output_tensor, dict):
        raise CompatibilityEvidenceError("Engine sidecar must describe one input and one output")

    input_name = _required_text(input_tensor.get("name"), "Engine input name")
    output_name = _required_text(output_tensor.get("name"), "Engine output name")
    tensorrt_version = _required_text(sidecar.get("tensorrt_version"), "Engine TensorRT version")
    precision = _required_text(sidecar.get("precision"), "Engine precision")
    io_precision = _required_text(sidecar.get("io_precision"), "Engine I/O precision")
    optimization_level = sidecar.get("builder_optimization_level")
    if not isinstance(optimization_level, int) or isinstance(optimization_level, bool):
        raise CompatibilityEvidenceError("Engine builder optimization level must be an integer")
    raw_builder_flags = sidecar.get("builder_flags")
    if not isinstance(raw_builder_flags, list) or any(
        not isinstance(flag, str) for flag in raw_builder_flags
    ):
        raise CompatibilityEvidenceError("Engine builder flags must be a list of strings")
    builder_flags = [public_value(flag, "Engine builder flag") for flag in raw_builder_flags]

    spec = make_upscale_model_spec(
        name="compatibility-candidate",
        input_name=input_name,
        output_name=output_name,
        input_shape=_shape(input_tensor.get("shape"), "Engine input shape"),
        output_shape=_shape(output_tensor.get("shape"), "Engine output shape"),
        input_dtype=_tensor_dtype(input_tensor.get("dtype"), "Engine input"),
        output_dtype=_tensor_dtype(output_tensor.get("dtype"), "Engine output"),
    )
    if sidecar.get("input_profile") is not None:
        raise CompatibilityEvidenceError("Compatibility reports require a static engine")
    if sidecar.get("preprocess_version") != "uint8_to_float_0_1":
        raise CompatibilityEvidenceError("Engine preprocess contract is unsupported")
    if sidecar.get("postprocess_version") != "float_0_1_to_uint8":
        raise CompatibilityEvidenceError("Engine postprocess contract is unsupported")

    summary = {
        "schema_version": 1,
        "engine": identity,
        "onnx_sha256": model_sha256,
        "tensorrt_version": tensorrt_version,
        "precision": precision,
        "io_precision": io_precision,
        "input": {
            "name": spec.inputs[0].name,
            "shape": list(spec.inputs[0].shape),
            "dtype": spec.inputs[0].dtype,
        },
        "output": {
            "name": spec.outputs[0].name,
            "shape": list(spec.outputs[0].shape),
            "dtype": spec.outputs[0].dtype,
        },
        "scale": spec.scale,
        "preprocess": spec.preprocess,
        "postprocess": spec.postprocess,
        "builder_optimization_level": optimization_level,
        "builder_flags": builder_flags,
    }
    return summary, spec


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompatibilityEvidenceError(f"{label} must be numeric")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise CompatibilityEvidenceError(f"{label} must be finite")
    return number


def conformance_evidence(
    path: Path,
    *,
    source_identity: dict[str, Any],
    expected_scale: int,
) -> dict[str, Any]:
    """Validate and sanitize source-export conformance evidence."""
    report = load_json_object(path, "Export-conformance report")
    if report.get("document_type") != EXPORT_CONFORMANCE_DOCUMENT_TYPE:
        raise CompatibilityEvidenceError("Export-conformance document_type is invalid")
    if (
        report.get("schema_version") != EXPORT_CONFORMANCE_SCHEMA_VERSION
        or report.get("status") != "valid"
    ):
        raise CompatibilityEvidenceError("Export-conformance report is not valid schema v2")
    if report.get("export_contract") != EXPORT_CONTRACT_METADATA_VALUE:
        raise CompatibilityEvidenceError("Export-conformance contract is unsupported")

    model = report.get("model")
    if not isinstance(model, dict):
        raise CompatibilityEvidenceError("Export-conformance model identity is missing")
    if model.get("source_sha256") != source_identity["sha256"]:
        raise CompatibilityEvidenceError("Source artifact hash does not match export conformance")
    if model.get("source_size_bytes") != source_identity["size_bytes"]:
        raise CompatibilityEvidenceError("Source artifact size does not match export conformance")
    if model.get("scale") != expected_scale:
        raise CompatibilityEvidenceError("Export-conformance scale does not match the engine")
    model_name = _required_text(model.get("name"), "Export-conformance model name")

    probe = report.get("probe")
    expected_input_shape = [1, 3, EXPORT_PROBE_HEIGHT, EXPORT_PROBE_WIDTH]
    if not isinstance(probe, dict):
        raise CompatibilityEvidenceError("Export-conformance probe is missing")
    if probe.get("version") != EXPORT_PROBE_VERSION:
        raise CompatibilityEvidenceError("Export-conformance probe version is unsupported")
    if probe.get("input_shape") != expected_input_shape:
        raise CompatibilityEvidenceError("Export-conformance probe input shape is invalid")
    if probe.get("input_sha256") != EXPORT_PROBE_SHA256:
        raise CompatibilityEvidenceError("Export-conformance probe hash is invalid")
    output_shape = _shape(probe.get("output_shape"), "Export-conformance probe output shape")
    try:
        probe_scale = infer_upscale_scale(expected_input_shape, output_shape)
    except ExportConformanceError as exc:
        raise CompatibilityEvidenceError("Export-conformance probe scale is invalid") from exc
    if probe_scale != expected_scale:
        raise CompatibilityEvidenceError("Export-conformance probe scale does not match the engine")

    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise CompatibilityEvidenceError("Export-conformance comparison is missing")
    thresholds = comparison.get("thresholds")
    metrics = comparison.get("metrics")
    if not isinstance(thresholds, dict) or not isinstance(metrics, dict):
        raise CompatibilityEvidenceError("Export-conformance metrics are missing")
    if comparison.get("reference") != "pytorch-fp32":
        raise CompatibilityEvidenceError("Export-conformance reference is unsupported")
    if comparison.get("candidate") != "onnxruntime-cpu-fp32":
        raise CompatibilityEvidenceError("Export-conformance candidate is unsupported")
    canonical_thresholds = {
        "max_abs": EXPORT_MAX_ABS_THRESHOLD,
        "rmse": EXPORT_RMSE_THRESHOLD,
        "min_psnr_db": EXPORT_MIN_PSNR_DB,
    }
    if thresholds != canonical_thresholds:
        raise CompatibilityEvidenceError("Export-conformance thresholds are not canonical")
    sanitized_metrics = {
        name: _finite_number(metrics.get(name), f"Conformance {name}")
        for name in ("max_abs", "mean_abs", "rmse", "relative_l2", "psnr_db")
    }
    if (
        sanitized_metrics["max_abs"] > EXPORT_MAX_ABS_THRESHOLD
        or sanitized_metrics["rmse"] > EXPORT_RMSE_THRESHOLD
        or sanitized_metrics["psnr_db"] < EXPORT_MIN_PSNR_DB
    ):
        raise CompatibilityEvidenceError("Export-conformance metrics exceed their thresholds")

    tools = report.get("tools")
    if not isinstance(tools, dict):
        raise CompatibilityEvidenceError("Export-conformance tool versions are missing")
    sanitized_tools = {
        package: _required_text(tools.get(package), f"Export tool {package} version")
        for package in _EXPORT_TOOL_PACKAGES
    }

    exports = report.get("exports")
    if not isinstance(exports, list) or not exports:
        raise CompatibilityEvidenceError("Export-conformance ONNX identities are missing")
    export_identities = []
    for export in exports:
        if not isinstance(export, dict):
            raise CompatibilityEvidenceError("Export-conformance ONNX identity is invalid")
        size_bytes = export.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise CompatibilityEvidenceError("Exported ONNX size must be a positive integer")
        export_identities.append(
            {
                "name": _artifact_name(export.get("path"), "Exported ONNX name"),
                "sha256": _require_sha256(export.get("sha256"), "Exported ONNX SHA256"),
                "size_bytes": size_bytes,
            }
        )

    return {
        "document": file_identity(path),
        "status": "valid",
        "export_contract": EXPORT_CONTRACT_METADATA_VALUE,
        "model": {
            "name": model_name,
            "scale": expected_scale,
            "source_sha256": model.get("source_sha256"),
            "source_size_bytes": model.get("source_size_bytes"),
        },
        "probe": {
            "version": EXPORT_PROBE_VERSION,
            "input_shape": expected_input_shape,
            "input_sha256": EXPORT_PROBE_SHA256,
            "output_shape": list(output_shape),
        },
        "comparison": {
            "reference": "pytorch-fp32",
            "candidate": "onnxruntime-cpu-fp32",
            "thresholds": canonical_thresholds,
            "metrics": sanitized_metrics,
        },
        "tools": sanitized_tools,
        "exports": export_identities,
    }


def command_evidence(path: Path) -> str:
    """Load a bounded command log and reject common private or secret values."""
    try:
        if not path.is_file():
            raise CompatibilityEvidenceError(f"Command log does not exist: {path.name}")
        if path.stat().st_size > 16 * 1024:
            raise CompatibilityEvidenceError("Command log exceeds 16 KiB")
        commands = path.read_text(encoding="utf-8").strip()
    except CompatibilityEvidenceError:
        raise
    except (OSError, UnicodeError) as exc:
        raise CompatibilityEvidenceError("Command log must be UTF-8 text") from exc
    if not commands:
        raise CompatibilityEvidenceError("Command log is empty")
    if "\x00" in commands:
        raise CompatibilityEvidenceError("Command log contains NUL bytes")
    for pattern, label in _SENSITIVE_COMMAND_PATTERNS:
        if pattern.search(commands):
            raise CompatibilityEvidenceError(
                f"Command log contains a possible {label}; sanitize it before reporting"
            )
    return commands


def public_value(value: str, label: str) -> str:
    """Validate short public metadata before embedding it in an issue body."""
    normalized = value.strip()
    if not normalized:
        raise CompatibilityEvidenceError(f"{label} is required")
    if len(normalized) > 512 or "\n" in normalized or "\r" in normalized:
        raise CompatibilityEvidenceError(f"{label} must be one bounded line")
    for pattern, sensitive_label in _SENSITIVE_COMMAND_PATTERNS:
        if pattern.search(normalized):
            raise CompatibilityEvidenceError(
                f"{label} contains a possible {sensitive_label}; sanitize it before reporting"
            )
    return normalized
