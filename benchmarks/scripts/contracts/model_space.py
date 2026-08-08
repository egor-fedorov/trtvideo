"""Identity validation for tensor-space quality evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.scripts.contracts.manifest import ManifestContractError


@dataclass(frozen=True)
class TensorComparisonExpectation:
    """Expected identity of one external tensor capture."""

    implementation: str
    engine_sha256: str
    image_id: str
    repository_revision: str
    execution_profile: dict[str, Any]


@dataclass(frozen=True)
class TensorReportExpectation:
    """Expected shared identity of a tensor-space report."""

    workload_id: str
    variant: str
    contract_version: int
    input_sha256: str
    onnx_sha256: str
    comparisons: tuple[TensorComparisonExpectation, ...]
    execution_profile: str | None = None
    frame_indices: list[int] | None = None
    reference_engine_sha256: str | None = None
    reference_image_id: str | None = None
    reference_revision: str | None = None
    reference_source_dirty: str | None = None
    reference_execution_profile: dict[str, Any] | None = None


def validate_inference_report(
    report: dict[str, Any],
    *,
    expectation: TensorReportExpectation,
) -> dict[str, dict[str, Any]]:
    """Validate mandatory shared-input TensorRT parity evidence."""
    comparisons = _validate_report(
        report,
        expectation=expectation,
        document_type="inference-parity",
        status="valid",
        acceptance_gate=True,
        label="Inference parity",
    )
    reference_hash = report.get("reference", {}).get("capture_manifest_sha256")
    if report.get("assets", {}).get("canonical_input_manifest_sha256") != reference_hash:
        raise ManifestContractError("Inference parity canonical input source changed")
    for implementation, comparison in comparisons.items():
        if comparison.get("canonical_input_manifest_sha256") != reference_hash:
            raise ManifestContractError(
                f"Inference parity {implementation} canonical input source changed"
            )
    return comparisons


def validate_preprocessing_report(
    report: dict[str, Any],
    *,
    expectation: TensorReportExpectation,
) -> dict[str, dict[str, Any]]:
    """Validate identity and completeness of non-gating preprocessing evidence."""
    return _validate_report(
        report,
        expectation=expectation,
        document_type="preprocessing-diagnostic",
        status="complete",
        acceptance_gate=False,
        label="Preprocessing diagnostic",
    )


def _validate_report(
    report: dict[str, Any],
    *,
    expectation: TensorReportExpectation,
    document_type: str,
    status: str,
    acceptance_gate: bool,
    label: str,
) -> dict[str, dict[str, Any]]:
    checks = {
        "document type": (report.get("document_type"), document_type),
        "status": (report.get("status"), status),
        "publishable": (report.get("publishable"), True),
        "acceptance role": (report.get("acceptance_gate"), acceptance_gate),
        "contract version": (report.get("contract_version"), expectation.contract_version),
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
        "frame indices": (report.get("frame_indices"), expectation.frame_indices),
        "reference engine": (
            report.get("reference", {}).get("engine_sha256"),
            expectation.reference_engine_sha256,
        ),
        "reference image": (
            report.get("reference", {}).get("image", {}).get("id"),
            expectation.reference_image_id,
        ),
        "reference revision": (
            report.get("reference", {}).get("image", {}).get("repository_revision"),
            expectation.reference_revision,
        ),
        "reference source state": (
            str(report.get("reference", {}).get("image", {}).get("source_dirty")),
            expectation.reference_source_dirty,
        ),
        "reference execution profile": (
            report.get("reference", {}).get("execution_profile"),
            expectation.reference_execution_profile,
        ),
    }
    for check_label, (actual, expected) in checks.items():
        if expected is not None and actual != expected:
            raise ManifestContractError(f"{label} report changed {check_label}")
    _validate_capture_hash(report.get("reference"), label=f"{label} reference")

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ManifestContractError(f"{label} report has no comparisons")
    by_implementation: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        implementation = comparison.get("implementation")
        if isinstance(implementation, str):
            by_implementation[implementation] = comparison
    expected_names = {candidate.implementation for candidate in expectation.comparisons}
    if set(by_implementation) != expected_names or len(comparisons) != len(expected_names):
        raise ManifestContractError(f"{label} report comparison set changed")
    for candidate in expectation.comparisons:
        comparison = by_implementation[candidate.implementation]
        candidate_checks = {
            "status": (comparison.get("status"), status),
            "engine": (comparison.get("engine_sha256"), candidate.engine_sha256),
            "execution profile": (
                comparison.get("execution_profile"),
                candidate.execution_profile,
            ),
            "image": (comparison.get("image", {}).get("id"), candidate.image_id),
            "revision": (
                comparison.get("image", {}).get("repository_revision"),
                candidate.repository_revision,
            ),
            "source state": (
                str(comparison.get("image", {}).get("source_dirty")),
                "0",
            ),
        }
        for check_label, (actual, expected) in candidate_checks.items():
            if actual != expected:
                raise ManifestContractError(
                    f"{label} report changed {candidate.implementation} {check_label}"
                )
        _validate_capture_hash(
            comparison,
            label=f"{label} {candidate.implementation}",
        )
    return by_implementation


def _validate_capture_hash(value: Any, *, label: str) -> None:
    checksum = value.get("capture_manifest_sha256") if isinstance(value, dict) else None
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ManifestContractError(f"{label} has no valid capture manifest SHA256")
