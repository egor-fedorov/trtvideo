"""Validate tuned sweep evidence and select one winner per competitor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.manifest import (
    ManifestContractError,
    RunExpectation,
    RunIdentity,
    artifact_path,
    execution_profile,
    load_json,
    validate_run_manifest,
)
from benchmarks.scripts.contracts.model_space import (
    ModelSpaceComparisonExpectation,
    ModelSpaceReportExpectation,
    validate_model_space_report,
)
from benchmarks.scripts.tuning.contract import (
    TunedCandidate,
    TuningContract,
    TuningContractError,
    load_tuning_contract,
)
from benchmarks.scripts.workloads.manifest import load_manifest

PRODUCTS = {
    "vstrt": "vs-mlrt",
    "vsgan": "VSGAN-tensorrt-docker",
}


class TuningEvidenceError(RuntimeError):
    """Raised when tuned evidence cannot be read safely."""


def _positive_float(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TuningEvidenceError(f"{label} must be a positive finite number")
    return float(value)


def _non_negative_float(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise TuningEvidenceError(f"{label} must be a non-negative finite number")
    return float(value)


@dataclass
class CandidateAssessment:
    """Eligibility and ranking data for one declared candidate."""

    candidate: TunedCandidate
    evidence_complete: bool = True
    errors: list[str] = field(default_factory=list)
    median_fps: float | None = None
    relative_spread: float | None = None
    identity: RunIdentity | None = None
    suite_path: str | None = None
    model_space_report_path: str | None = None

    @property
    def eligible(self) -> bool:
        return self.evidence_complete and not self.errors

    def reject(self, error: str, *, evidence_complete: bool = True) -> None:
        if error not in self.errors:
            self.errors.append(error)
        if not evidence_complete:
            self.evidence_complete = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "implementation": self.candidate.implementation,
            "execution_profile": self.candidate.execution_profile(),
            "runner_arguments": self.candidate.runner_arguments(),
            "status": "eligible" if self.eligible else "disqualified",
            "evidence_complete": self.evidence_complete,
            "median_fps": self.median_fps,
            "relative_spread": self.relative_spread,
            "identity": self.identity.as_dict() if self.identity is not None else None,
            "evidence": {
                "suite": self.suite_path,
                "model_space_report": self.model_space_report_path,
            },
            "errors": self.errors,
        }


def candidate_directory(sweep_dir: Path, candidate: TunedCandidate) -> Path:
    """Return the collision-free root for one candidate."""
    return (
        sweep_dir
        / "candidates"
        / candidate.implementation
        / candidate.candidate_id
    )


def load_disqualifications(
    path: Path | None,
    *,
    contract: TuningContract,
) -> dict[str, dict[str, str]]:
    """Load candidate rejections produced by the full winner quality gate."""
    if path is None or not path.exists():
        return {}
    value = load_json(path)
    if value.get("schema_version") != 1:
        raise TuningEvidenceError("Unsupported disqualification schema version")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise TuningEvidenceError("Disqualifications entries must be an array")
    known_ids = {candidate.candidate_id for candidate in contract.candidates}
    result: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TuningEvidenceError("Disqualification entry must be an object")
        candidate_id = entry.get("candidate_id")
        reason = entry.get("reason")
        evidence = entry.get("evidence")
        if candidate_id not in known_ids:
            raise TuningEvidenceError(
                f"Disqualification references unknown candidate: {candidate_id!r}"
            )
        if not isinstance(reason, str) or not reason:
            raise TuningEvidenceError(
                f"Disqualification {candidate_id} has no reason"
            )
        if not isinstance(evidence, str) or not evidence:
            raise TuningEvidenceError(
                f"Disqualification {candidate_id} has no evidence path"
            )
        result[str(candidate_id)] = {
            "reason": reason,
            "evidence": evidence,
        }
    return result


def _validate_suite(
    assessment: CandidateAssessment,
    *,
    root: Path,
    suite_path: Path,
    workload_id: str,
    variant: str,
    contract_version: int,
    max_relative_spread: float,
) -> None:
    assessment.suite_path = suite_path.relative_to(root).as_posix()
    if not suite_path.is_file():
        assessment.reject("Performance suite evidence is missing", evidence_complete=False)
        return
    try:
        suite = load_json(suite_path)
        if suite.get("status") != "valid":
            assessment.reject(f"Performance suite status is {suite.get('status')!r}")
        if suite.get("workload_id") != workload_id or suite.get("variant") != variant:
            assessment.reject("Performance suite changed workload or variant")
        if suite.get("benchmark_contract_version") != contract_version:
            assessment.reject("Performance suite changed benchmark contract version")
        parameters = suite.get("parameters")
        if not isinstance(parameters, dict):
            raise TuningEvidenceError("Performance suite has no parameters")
        if execution_profile(parameters) != assessment.candidate.execution_profile():
            assessment.reject("Performance suite execution profile changed")
        statistics = suite.get("statistics")
        if not isinstance(statistics, dict):
            raise TuningEvidenceError("Performance suite has no statistics")
        assessment.median_fps = _positive_float(
            statistics.get("median_fps"),
            label="Performance median FPS",
        )
        assessment.relative_spread = _non_negative_float(
            statistics.get("relative_spread"),
            label="Performance relative spread",
        )
        if assessment.relative_spread > max_relative_spread:
            assessment.reject(
                "Performance spread exceeds the tuning selection contract "
                f"({assessment.relative_spread:.2%} > {max_relative_spread:.2%})"
            )
        runs = suite.get("runs")
        if not isinstance(runs, list) or not runs:
            raise TuningEvidenceError("Performance suite has no run manifests")
        identities = []
        for run in runs:
            if not isinstance(run, dict):
                raise TuningEvidenceError("Performance suite run entry is invalid")
            manifest_path = artifact_path(
                root,
                run.get("manifest"),
                label="Performance run",
            )
            if not manifest_path.is_file():
                raise TuningEvidenceError(
                    f"Performance run manifest is missing: {manifest_path}"
                )
            identities.append(
                validate_run_manifest(
                    load_json(manifest_path),
                    expectation=RunExpectation(
                        product=PRODUCTS[assessment.candidate.implementation],
                        workload_id=workload_id,
                        variant=variant,
                        benchmark_contract_version=contract_version,
                        implementation=assessment.candidate.implementation,
                        execution_profile=assessment.candidate.execution_profile(),
                        require_media_validation=True,
                    ),
                    checksum_length=64,
                )
            )
        first_identity = identities[0]
        if any(
            identity.implementation_key() != first_identity.implementation_key()
            for identity in identities[1:]
        ):
            assessment.reject("Performance runs changed immutable evidence")
        assessment.identity = first_identity
    except (ManifestContractError, TuningEvidenceError) as exc:
        assessment.reject(str(exc), evidence_complete=False)


def _validate_model_space(
    assessment: CandidateAssessment,
    *,
    root: Path,
    report_path: Path,
    workload_id: str,
    variant: str,
) -> None:
    assessment.model_space_report_path = report_path.relative_to(root).as_posix()
    if not report_path.is_file():
        assessment.reject("Model-space evidence is missing", evidence_complete=False)
        return
    if assessment.identity is None:
        assessment.reject(
            "Model-space identity cannot be checked without a valid run",
            evidence_complete=False,
        )
        return
    try:
        identity = assessment.identity
        report = load_json(report_path)
    except ManifestContractError as exc:
        assessment.reject(str(exc), evidence_complete=False)
        return
    try:
        validate_model_space_report(
            report,
            expectation=ModelSpaceReportExpectation(
                workload_id=workload_id,
                variant=variant,
                input_sha256=identity.input_sha256,
                onnx_sha256=identity.onnx_sha256,
                comparisons=(
                    ModelSpaceComparisonExpectation(
                        implementation=PRODUCTS[
                            assessment.candidate.implementation
                        ],
                        engine_sha256=identity.engine_sha256,
                        image_id=identity.image_id,
                        repository_revision=identity.repository_revision,
                        execution_profile=assessment.candidate.execution_profile(),
                        comparison_class="tuned",
                    ),
                ),
            ),
        )
    except (ManifestContractError, TuningEvidenceError) as exc:
        assessment.reject(str(exc))


def _enforce_shared_contract(assessments: list[CandidateAssessment]) -> None:
    identities = [
        assessment.identity
        for assessment in assessments
        if assessment.identity is not None
    ]
    if not identities:
        return
    shared_key = identities[0].shared_model_key()
    for assessment in assessments:
        identity = assessment.identity
        if identity is not None and identity.shared_model_key() != shared_key:
            assessment.reject(
                "Candidate changed shared workload, model, encoder, or revision"
            )
    for implementation in PRODUCTS:
        implementation_assessments = [
            assessment
            for assessment in assessments
            if assessment.candidate.implementation == implementation
            and assessment.identity is not None
        ]
        if not implementation_assessments:
            continue
        implementation_key = implementation_assessments[0].identity
        assert implementation_key is not None
        expected = implementation_key.implementation_key()
        for assessment in implementation_assessments:
            identity = assessment.identity
            assert identity is not None
            if identity.implementation_key() != expected:
                assessment.reject(
                    "Candidate changed implementation image or engine"
                )


def _validate_disqualification(
    assessment: CandidateAssessment,
    rejection: dict[str, str],
    *,
    root: Path,
    workload_id: str,
    variant: str,
    contract_version: int,
) -> None:
    evidence_path = artifact_path(
        root,
        rejection["evidence"],
        label=f"{assessment.candidate.candidate_id} disqualification",
    )
    if not evidence_path.is_file():
        raise TuningEvidenceError(
            f"Disqualification evidence is missing: {evidence_path}"
        )
    report = load_json(evidence_path)
    if report.get("document_type") not in {
        "model-space-parity",
        "product-output-parity",
    }:
        raise TuningEvidenceError(
            "Candidate disqualification is not a full quality report"
        )
    if report.get("status") != "invalid":
        raise TuningEvidenceError(
            "Candidate disqualification must reference an invalid quality report"
        )
    if report.get("workload_id") != workload_id or report.get("variant") != variant:
        raise TuningEvidenceError(
            "Candidate disqualification changed workload or variant"
        )
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise TuningEvidenceError(
            "Candidate disqualification report has no comparisons"
        )
    product = PRODUCTS[assessment.candidate.implementation]
    matching = [
        comparison
        for comparison in comparisons
        if isinstance(comparison, dict)
        and comparison.get("implementation") == product
        and comparison.get("status") == "invalid"
    ]
    if len(matching) != 1:
        raise TuningEvidenceError(
            "Candidate disqualification does not prove one matching quality failure"
        )
    comparison = matching[0]
    profile = comparison.get("execution_profile")
    if profile is not None and profile != assessment.candidate.execution_profile():
        raise TuningEvidenceError(
            "Candidate disqualification execution profile changed"
        )
    run_manifest_value = comparison.get("run_manifest")
    if run_manifest_value is not None:
        run_manifest_path = artifact_path(
            root,
            run_manifest_value,
            label="Disqualified product-output run",
        )
        identity = validate_run_manifest(
            load_json(run_manifest_path),
            expectation=RunExpectation(
                product=PRODUCTS[assessment.candidate.implementation],
                workload_id=workload_id,
                variant=variant,
                benchmark_contract_version=contract_version,
                implementation=assessment.candidate.implementation,
                execution_profile=assessment.candidate.execution_profile(),
                require_media_validation=True,
            ),
            checksum_length=64,
        )
        if (
            assessment.identity is not None
            and identity.implementation_key()
            != assessment.identity.implementation_key()
        ):
            raise TuningEvidenceError(
                "Candidate disqualification changed implementation evidence"
            )
    assessment.reject(
        "Full winner quality gate failed: " + rejection["reason"]
    )


def _winner(
    assessments: list[CandidateAssessment],
    implementation: str,
) -> CandidateAssessment | None:
    eligible = [
        assessment
        for assessment in assessments
        if assessment.candidate.implementation == implementation
        and assessment.eligible
        and assessment.median_fps is not None
    ]
    if not eligible:
        return None

    def ranking_key(assessment: CandidateAssessment) -> tuple[float, str]:
        median_fps = assessment.median_fps
        assert median_fps is not None
        return (-median_fps, assessment.candidate.candidate_id)

    return sorted(
        eligible,
        key=ranking_key,
    )[0]


def rank_tuned_candidates(
    *,
    contract: TuningContract,
    workload: dict[str, Any],
    variant: str,
    sweep_dir: Path,
    root: Path,
    disqualifications: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate all declared candidates and create an immutable selection report."""
    root = root.resolve()
    sweep_dir = sweep_dir.resolve()
    contract_version = int(workload["benchmark"]["contract_version"])
    assessments = []
    for candidate in contract.candidates:
        assessment = CandidateAssessment(candidate=candidate)
        directory = candidate_directory(sweep_dir, candidate)
        _validate_suite(
            assessment,
            root=root,
            suite_path=directory / "performance" / "suite.json",
            workload_id=workload["id"],
            variant=variant,
            contract_version=contract_version,
            max_relative_spread=contract.selection.max_relative_spread,
        )
        _validate_model_space(
            assessment,
            root=root,
            report_path=directory / "model-space-parity.json",
            workload_id=workload["id"],
            variant=variant,
        )
        assessments.append(assessment)
    _enforce_shared_contract(assessments)
    disqualifications = disqualifications or {}
    for assessment in assessments:
        rejection = disqualifications.get(assessment.candidate.candidate_id)
        if rejection is not None:
            _validate_disqualification(
                assessment,
                rejection,
                root=root,
                workload_id=workload["id"],
                variant=variant,
                contract_version=contract_version,
            )

    winners = {
        implementation: _winner(assessments, implementation)
        for implementation in PRODUCTS
    }
    incomplete = [
        assessment.candidate.candidate_id
        for assessment in assessments
        if not assessment.evidence_complete
    ]
    errors = []
    if contract.selection.require_complete_sweep and incomplete:
        errors.append(
            "Complete evidence is missing for declared candidates: "
            + ", ".join(incomplete)
        )
    for implementation, winner in winners.items():
        if winner is None:
            errors.append(f"No eligible {implementation} candidate remains")

    first_identity = next(
        (
            assessment.identity
            for assessment in assessments
            if assessment.identity is not None
        ),
        None,
    )
    report = {
        "schema_version": 1,
        "document_type": "tuned-candidate-selection",
        "status": "valid" if not errors else "invalid",
        "publishable": False,
        "scope": "tuning-selection",
        "workload_id": workload["id"],
        "variant": variant,
        "benchmark_contract_version": contract_version,
        "selection_policy": contract.selection.as_dict(),
        "project_profile": contract.project_profile.as_dict(),
        "disqualifications": disqualifications,
        "environment": {
            "repository_revision": (
                first_identity.repository_revision
                if first_identity is not None
                else None
            )
        },
        "candidates": [assessment.as_dict() for assessment in assessments],
        "winners": {
            implementation: (
                {
                    "candidate_id": winner.candidate.candidate_id,
                    "median_fps": winner.median_fps,
                    "relative_spread": winner.relative_spread,
                    "execution_profile": winner.candidate.execution_profile(),
                    "runner_arguments": winner.candidate.runner_arguments(),
                }
                if winner is not None
                else None
            )
            for implementation, winner in winners.items()
        },
        "errors": errors,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="benchmarks/tuning/candidates.json",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], required=True)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--disqualifications",
        default=None,
        help="Optional full-quality candidate rejection document",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        root = Path(args.root).resolve()
        contract = load_tuning_contract(Path(args.contract))
        report = rank_tuned_candidates(
            contract=contract,
            workload=load_manifest(Path(args.manifest)),
            variant=args.variant,
            sweep_dir=Path(args.sweep_dir),
            root=root,
            disqualifications=load_disqualifications(
                Path(args.disqualifications) if args.disqualifications else None,
                contract=contract,
            ),
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (
        ManifestContractError,
        TuningContractError,
        TuningEvidenceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Tuned candidate selection {report['status']}: {output_path}")
    if report["status"] != "valid":
        sys.exit(2)


if __name__ == "__main__":
    main()
