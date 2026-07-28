from __future__ import annotations

from typing import Any

import pytest

from benchmarks.scripts.contracts.manifest import ManifestContractError
from benchmarks.scripts.contracts.model_space import (
    ModelSpaceComparisonExpectation,
    ModelSpaceReportExpectation,
    validate_model_space_report,
)


def test_model_space_contract_rejects_duplicate_implementation() -> None:
    comparison: dict[str, Any] = {
        "implementation": "vs-mlrt",
        "status": "valid",
        "engine_sha256": "engine",
        "execution_profile": {"execution_profile": "tuned"},
        "image": {
            "id": "image",
            "repository_revision": "revision",
            "source_dirty": "0",
        },
    }
    report = {
        "document_type": "model-space-parity",
        "status": "valid",
        "publishable": True,
        "workload_id": "workload-v1",
        "variant": "1080p",
        "assets": {
            "input_sha256": "input",
            "onnx_sha256": "onnx",
        },
        "comparisons": [comparison, dict(comparison)],
    }
    expectation = ModelSpaceReportExpectation(
        workload_id="workload-v1",
        variant="1080p",
        input_sha256="input",
        onnx_sha256="onnx",
        comparisons=(
            ModelSpaceComparisonExpectation(
                implementation="vs-mlrt",
                engine_sha256="engine",
                image_id="image",
                repository_revision="revision",
                execution_profile={"execution_profile": "tuned"},
            ),
        ),
    )

    with pytest.raises(
        ManifestContractError,
        match="comparison set changed",
    ):
        validate_model_space_report(report, expectation=expectation)
