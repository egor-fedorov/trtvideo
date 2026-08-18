from __future__ import annotations

import copy
from pathlib import Path

import pytest

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
    conformance_report_path,
    infer_upscale_scale,
    validate_conformance_report,
)

TOOLS = {
    "torch": "2.11.0",
    "onnx": "1.20.1",
    "onnxruntime": "1.23.2",
    "onnxscript": "0.5.7",
    "spandrel": "0.4.1",
}
EXPORTS = {"model_720p.onnx": ("b" * 64, 1234)}


def _valid_report(*, scale: int = 2) -> dict:
    return {
        "document_type": EXPORT_CONFORMANCE_DOCUMENT_TYPE,
        "schema_version": EXPORT_CONFORMANCE_SCHEMA_VERSION,
        "status": "valid",
        "export_contract": EXPORT_CONTRACT_METADATA_VALUE,
        "model": {
            "name": "model",
            "scale": scale,
            "source_sha256": "a" * 64,
            "source_size_bytes": 42,
        },
        "probe": {
            "version": EXPORT_PROBE_VERSION,
            "input_shape": [1, 3, EXPORT_PROBE_HEIGHT, EXPORT_PROBE_WIDTH],
            "input_sha256": EXPORT_PROBE_SHA256,
            "output_shape": [
                1,
                3,
                EXPORT_PROBE_HEIGHT * scale,
                EXPORT_PROBE_WIDTH * scale,
            ],
        },
        "comparison": {
            "reference": "pytorch-fp32",
            "candidate": "onnxruntime-cpu-fp32",
            "thresholds": {
                "max_abs": EXPORT_MAX_ABS_THRESHOLD,
                "rmse": EXPORT_RMSE_THRESHOLD,
                "min_psnr_db": EXPORT_MIN_PSNR_DB,
            },
            "metrics": {
                "max_abs": EXPORT_MAX_ABS_THRESHOLD / 2,
                "mean_abs": EXPORT_RMSE_THRESHOLD / 2,
                "rmse": EXPORT_RMSE_THRESHOLD / 2,
                "relative_l2": 1e-6,
                "psnr_db": EXPORT_MIN_PSNR_DB + 10,
            },
        },
        "tools": TOOLS,
        "exports": [
            {
                "path": "model_720p.onnx",
                "sha256": "b" * 64,
                "size_bytes": 1234,
            }
        ],
    }


def _validate(report: dict) -> dict:
    return validate_conformance_report(
        report,
        model_name="model",
        source_sha256="a" * 64,
        source_size_bytes=42,
        exported_files=EXPORTS,
        tool_versions=TOOLS,
    )


def test_conformance_report_path_is_model_scoped(tmp_path: Path) -> None:
    assert conformance_report_path(tmp_path, "span") == (tmp_path / "span.export-conformance.json")


def test_validate_conformance_report_accepts_bound_evidence() -> None:
    comparison = _validate(_valid_report())

    assert comparison["metrics"]["psnr_db"] == EXPORT_MIN_PSNR_DB + 10


def test_validate_conformance_report_accepts_inferred_non_x2_scale() -> None:
    report = _valid_report(scale=4)

    comparison = validate_conformance_report(
        report,
        model_name="model",
        source_sha256="a" * 64,
        source_size_bytes=42,
        exported_files=EXPORTS,
        tool_versions=TOOLS,
        expected_scale=4,
    )

    assert comparison["metrics"]["psnr_db"] == EXPORT_MIN_PSNR_DB + 10


def test_validate_conformance_report_rejects_unexpected_scale() -> None:
    with pytest.raises(ExportConformanceError, match="scale must be 2x"):
        validate_conformance_report(
            _valid_report(scale=4),
            model_name="model",
            source_sha256="a" * 64,
            source_size_bytes=42,
            exported_files=EXPORTS,
            tool_versions=TOOLS,
            expected_scale=2,
        )


def test_validate_conformance_report_rejects_scale_shape_disagreement() -> None:
    report = _valid_report(scale=4)
    report["probe"]["output_shape"] = [
        1,
        3,
        EXPORT_PROBE_HEIGHT * 2,
        EXPORT_PROBE_WIDTH * 2,
    ]

    with pytest.raises(ExportConformanceError, match="does not match its model scale"):
        _validate(report)


def test_validate_conformance_report_rejects_missing_scale() -> None:
    report = _valid_report()
    del report["model"]["scale"]

    with pytest.raises(ExportConformanceError, match="model scale is invalid"):
        _validate(report)


def test_infer_upscale_scale_rejects_non_uniform_output() -> None:
    with pytest.raises(ExportConformanceError, match="scale must be uniform"):
        infer_upscale_scale([1, 3, 16, 16], [1, 3, 48, 64])


def test_validate_conformance_report_rejects_relaxed_threshold() -> None:
    report = copy.deepcopy(_valid_report())
    report["comparison"]["thresholds"]["max_abs"] = 1.0

    with pytest.raises(ExportConformanceError, match="thresholds do not match"):
        _validate(report)


def test_validate_conformance_report_rejects_failed_metrics() -> None:
    report = copy.deepcopy(_valid_report())
    report["comparison"]["metrics"]["max_abs"] = EXPORT_MAX_ABS_THRESHOLD * 2

    with pytest.raises(ExportConformanceError, match="max_abs exceeds"):
        _validate(report)


def test_validate_conformance_report_rejects_stale_export_identity() -> None:
    report = _valid_report()

    with pytest.raises(ExportConformanceError, match="ONNX identities do not match"):
        validate_conformance_report(
            report,
            model_name="model",
            source_sha256="a" * 64,
            source_size_bytes=42,
            exported_files={"model_720p.onnx": ("d" * 64, 1234)},
            tool_versions=TOOLS,
        )


def test_validate_conformance_report_rejects_changed_toolchain() -> None:
    report = _valid_report()

    with pytest.raises(ExportConformanceError, match="tool versions do not match"):
        validate_conformance_report(
            report,
            model_name="model",
            source_sha256="a" * 64,
            source_size_bytes=42,
            exported_files=EXPORTS,
            tool_versions={**TOOLS, "torch": "2.12.0"},
        )
