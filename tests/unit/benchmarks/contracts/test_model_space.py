from __future__ import annotations

from typing import Any

import pytest

from benchmarks.scripts.contracts.manifest import ManifestContractError
from benchmarks.scripts.contracts.model_space import (
    TensorComparisonExpectation,
    TensorReportExpectation,
    validate_inference_report,
    validate_preprocessing_report,
)


def _expectation() -> TensorReportExpectation:
    return TensorReportExpectation(
        workload_id="workload-v1",
        variant="1080p",
        contract_version=3,
        input_sha256="input",
        onnx_sha256="onnx",
        comparisons=(
            TensorComparisonExpectation(
                implementation="vs-mlrt",
                engine_sha256="engine",
                image_id="image",
                repository_revision="revision",
                execution_profile={"execution_profile": "tuned"},
            ),
        ),
    )


def _comparison(*, status: str) -> dict[str, Any]:
    return {
        "implementation": "vs-mlrt",
        "status": status,
        "capture_manifest_sha256": "c" * 64,
        "canonical_input_manifest_sha256": "r" * 64,
        "engine_sha256": "engine",
        "execution_profile": {"execution_profile": "tuned"},
        "image": {
            "id": "image",
            "repository_revision": "revision",
            "source_dirty": "0",
        },
    }


def _report(*, document_type: str, status: str, acceptance_gate: bool) -> dict[str, Any]:
    return {
        "document_type": document_type,
        "status": status,
        "publishable": True,
        "acceptance_gate": acceptance_gate,
        "contract_version": 3,
        "workload_id": "workload-v1",
        "variant": "1080p",
        "assets": {
            "input_sha256": "input",
            "onnx_sha256": "onnx",
            "canonical_input_manifest_sha256": "r" * 64,
        },
        "reference": {"capture_manifest_sha256": "r" * 64},
        "comparisons": [_comparison(status=status)],
    }


def test_inference_contract_rejects_duplicate_implementation() -> None:
    report = _report(
        document_type="inference-parity",
        status="valid",
        acceptance_gate=True,
    )
    report["comparisons"].append(dict(report["comparisons"][0]))

    with pytest.raises(ManifestContractError, match="comparison set changed"):
        validate_inference_report(report, expectation=_expectation())


def test_preprocessing_contract_requires_non_gating_role() -> None:
    report = _report(
        document_type="preprocessing-diagnostic",
        status="complete",
        acceptance_gate=True,
    )

    with pytest.raises(ManifestContractError, match="acceptance role"):
        validate_preprocessing_report(report, expectation=_expectation())


def test_inference_contract_rejects_changed_shared_input_source() -> None:
    report = _report(
        document_type="inference-parity",
        status="valid",
        acceptance_gate=True,
    )
    report["comparisons"][0]["canonical_input_manifest_sha256"] = "x" * 64

    with pytest.raises(ManifestContractError, match="canonical input source"):
        validate_inference_report(report, expectation=_expectation())
