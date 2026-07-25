"""Shared identity validation for model-space quality reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.scripts.contracts.manifest import ManifestContractError


@dataclass(frozen=True)
class ModelSpaceComparisonExpectation:
    """Expected identity of one model-space candidate comparison."""

    implementation: str
    engine_sha256: str
    image_id: str
    repository_revision: str
    execution_profile: dict[str, Any]
    comparison_class: str | None = None


@dataclass(frozen=True)
class ModelSpaceReportExpectation:
    """Expected shared identity of a model-space report."""

    workload_id: str
    variant: str
    input_sha256: str
    onnx_sha256: str
    comparisons: tuple[ModelSpaceComparisonExpectation, ...]
    execution_profile: str | None = None
    frame_indices: list[int] | None = None
    reference_engine_sha256: str | None = None
    reference_image_id: str | None = None
    reference_revision: str | None = None
    reference_source_dirty: str | None = None
    reference_execution_profile: dict[str, Any] | None = None


def validate_model_space_report(
    report: dict[str, Any],
    *,
    expectation: ModelSpaceReportExpectation,
) -> dict[str, dict[str, Any]]:
    """Validate report header and every expected candidate identity."""
    checks = {
        "document type": (report.get("document_type"), "model-space-parity"),
        "status": (report.get("status"), "valid"),
        "publishable": (report.get("publishable"), True),
        "workload": (report.get("workload_id"), expectation.workload_id),
        "variant": (report.get("variant"), expectation.variant),
        "input SHA256": (
            report.get("assets", {}).get("input_sha256"),
            expectation.input_sha256,
        ),
        "ONNX SHA256": (
            report.get("assets", {}).get("onnx_sha256"),
            expectation.onnx_sha256,
        ),
        "execution profile": (
            report.get("execution_profile"),
            expectation.execution_profile,
        ),
        "frame indices": (
            report.get("frame_indices"),
            expectation.frame_indices,
        ),
        "reference engine": (
            report.get("reference", {}).get("engine_sha256"),
            expectation.reference_engine_sha256,
        ),
        "reference image": (
            report.get("reference", {}).get("image", {}).get("id"),
            expectation.reference_image_id,
        ),
        "reference revision": (
            report.get("reference", {})
            .get("image", {})
            .get("repository_revision"),
            expectation.reference_revision,
        ),
        "reference source state": (
            str(
                report.get("reference", {})
                .get("image", {})
                .get("source_dirty")
            ),
            expectation.reference_source_dirty,
        ),
        "reference execution profile": (
            report.get("reference", {}).get("execution_profile"),
            expectation.reference_execution_profile,
        ),
    }
    for label, (actual, expected) in checks.items():
        if expected is not None and actual != expected:
            raise ManifestContractError(f"Model-space report changed {label}")

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ManifestContractError("Model-space report has no comparisons")
    by_implementation: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        implementation = comparison.get("implementation")
        if isinstance(implementation, str):
            by_implementation[implementation] = comparison
    expected_names = {
        candidate.implementation for candidate in expectation.comparisons
    }
    if (
        set(by_implementation) != expected_names
        or len(comparisons) != len(expected_names)
    ):
        raise ManifestContractError(
            "Model-space report comparison set changed"
        )
    for candidate in expectation.comparisons:
        comparison = by_implementation[candidate.implementation]
        candidate_checks = {
            "status": (comparison.get("status"), "valid"),
            "engine": (
                comparison.get("engine_sha256"),
                candidate.engine_sha256,
            ),
            "execution profile": (
                comparison.get("execution_profile"),
                candidate.execution_profile,
            ),
            "comparison class": (
                comparison.get("comparison_class"),
                candidate.comparison_class,
            ),
            "image": (
                comparison.get("image", {}).get("id"),
                candidate.image_id,
            ),
            "revision": (
                comparison.get("image", {}).get("repository_revision"),
                candidate.repository_revision,
            ),
            "source state": (
                str(comparison.get("image", {}).get("source_dirty")),
                "0",
            ),
        }
        for label, (actual, expected) in candidate_checks.items():
            if expected is not None and actual != expected:
                raise ManifestContractError(
                    "Model-space report changed "
                    f"{candidate.implementation} {label}"
                )
    return by_implementation
