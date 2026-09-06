"""Validate tuned sweep evidence and select one winner per competitor."""

from __future__ import annotations

import argparse
import hashlib
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
from benchmarks.scripts.quality.preflight_tas import PreflightError, load_preflight_report
from benchmarks.scripts.runtime.suite import compute_suite_statistics, should_extend_suite
from benchmarks.scripts.tuning.adaptive import (
    CandidatePoint,
    has_confirmed_decline,
    resource_medians,
    resource_order,
    select_peak_equivalent,
    sentinel_recovers,
    shortlist,
    stream_count,
    upper_boundary_unresolved,
)
from benchmarks.scripts.tuning.contract import (
    Candidate,
    ImplementationProfile,
    MeasurementPolicy,
    TasCandidate,
    TunedCandidate,
    TuningContract,
    TuningContractError,
    load_tuning_contract,
)
from benchmarks.scripts.tuning.resource_limit import (
    ResourceLimitError,
    validate_cuda_oom_record,
)
from benchmarks.scripts.workloads.manifest import load_manifest

PRODUCTS = {
    "vstrt": "vs-mlrt",
    "tas": "TheAnimeScripter",
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

    candidate: Candidate
    evidence_complete: bool = True
    errors: list[str] = field(default_factory=list)
    median_fps: float | None = None
    relative_spread: float | None = None
    identity: RunIdentity | None = None
    suite_path: str | None = None
    median_peak_vram_mib: float | None = None
    median_cpu_cores: float | None = None

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
            **(
                {
                    "median_peak_vram_mib": self.median_peak_vram_mib,
                    "median_cpu_cores": self.median_cpu_cores,
                }
                if isinstance(self.candidate, TasCandidate)
                else {}
            ),
            "identity": self.identity.as_dict() if self.identity is not None else None,
            "evidence": {
                "suite": self.suite_path,
            },
            "errors": self.errors,
        }


def candidate_directory(sweep_dir: Path, candidate: Candidate) -> Path:
    """Return the collision-free root for one candidate."""
    return sweep_dir / "candidates" / candidate.implementation / candidate.candidate_id


def _verified_quality_report(root: Path, record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TuningEvidenceError("TAS preflight is missing hashed quality evidence")
    path = artifact_path(root, record.get("path"), label="TAS preflight quality")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise TuningEvidenceError("TAS preflight quality evidence SHA256 changed")
    return load_json(path)


def load_tas_preflight(
    path: Path,
    *,
    root: Path,
    contract: TuningContract,
    workload: dict[str, Any],
    variant: str,
    engine_sha256: str | None = None,
    tas_engine_sha256: str | None = None,
    workload_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Consume the quality module's verified preflight, never infer failures from exit codes."""
    if not path.is_file():
        raise TuningEvidenceError(f"TAS preflight is missing; run preflight-tas-quality: {path}")
    try:
        report = load_preflight_report(path, root=root)
    except PreflightError as exc:
        raise TuningEvidenceError(f"TAS preflight evidence is invalid: {exc}") from exc
    identity = report["identity"]
    if identity["workload_id"] != workload["id"] or identity["variant"] != variant:
        raise TuningEvidenceError("TAS preflight changed workload or variant")
    for key, expected in (
        ("engine", engine_sha256),
        ("tas_engine", tas_engine_sha256),
        ("workload_manifest", workload_sha256),
    ):
        if expected is not None and identity["files"][key]["sha256"] != expected:
            raise TuningEvidenceError(f"TAS preflight {key} SHA256 changed")
    profiles = report["profiles"]
    if set(profiles) != {
        candidate.candidate_id for candidate in contract.for_implementation("tas")
    }:
        raise TuningEvidenceError("TAS preflight must cover all four I/O configurations")
    result = {}
    for candidate_id, entry in profiles.items():
        candidate = contract.candidate(candidate_id)
        if entry["execution_profile"] != candidate.execution_profile():
            raise TuningEvidenceError("TAS preflight changed candidate execution profile")
        if entry["status"] not in {"valid", "disqualified"}:
            raise TuningEvidenceError(
                "TAS preflight did not prove eligibility or a quality failure"
            )
        evidence = entry["evidence"]
        result[candidate_id] = {
            "status": "valid" if entry["status"] == "valid" else "invalid",
            "execution_profile": candidate.execution_profile(),
            "quality": {
                "inference_parity": evidence["inference"],
                "preprocessing_diagnostic": evidence["preprocessing"],
                **(
                    {"product_output": evidence["product_output"]}
                    if "product_output" in evidence
                    else {}
                ),
            },
        }
    if not any(entry["status"] == "valid" for entry in result.values()):
        raise TuningEvidenceError("No TAS configuration passed preflight quality")
    return result


def _preflight_product_identity(
    comparison: dict[str, Any],
    candidate: Candidate,
    root: Path,
    workload: dict[str, Any],
    variant: str,
) -> RunIdentity:
    manifest = _verified_quality_report(
        root,
        {
            "path": comparison.get("run_manifest"),
            "sha256": comparison.get("run_manifest_sha256"),
        },
    )
    return validate_run_manifest(
        manifest,
        expectation=RunExpectation(
            product=PRODUCTS["tas"],
            workload_id=workload["id"],
            variant=variant,
            benchmark_contract_version=int(workload["benchmark"]["contract_version"]),
            implementation="tas",
            execution_profile=candidate.execution_profile(),
            require_media_validation=True,
            require_hardware_environment=True,
        ),
        checksum_length=64,
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
            raise TuningEvidenceError(f"Disqualification {candidate_id} has no reason")
        if not isinstance(evidence, str) or not evidence:
            raise TuningEvidenceError(f"Disqualification {candidate_id} has no evidence path")
        result[str(candidate_id)] = {
            "reason": reason,
            "evidence": evidence,
        }
    return result


def _performance_run_fps(manifest: dict[str, Any], *, policy: MeasurementPolicy) -> float:
    parameters = manifest["parameters"]
    expected_parameters = {
        "frames": policy.measured_frames,
        "warmup_frames": policy.warmup_frames,
        "bitrate_validation": policy.bitrate_validation,
    }
    changed = [
        key for key, expected in expected_parameters.items() if parameters.get(key) != expected
    ]
    if changed:
        raise TuningEvidenceError("Performance run changed stage parameters: " + ", ".join(changed))
    measured = manifest.get("measured")
    metrics = measured.get("metrics") if isinstance(measured, dict) else None
    if not isinstance(metrics, dict):
        raise TuningEvidenceError("Performance run has no measured metrics")
    if metrics.get("processed_frames") != policy.measured_frames:
        raise TuningEvidenceError("Performance run processed frame count changed")
    fps = _positive_float(metrics.get("end_to_end_fps"), label="Performance run FPS")
    wall_time = _positive_float(metrics.get("wall_time_sec"), label="Performance run wall time")
    if not math.isclose(fps, policy.measured_frames / wall_time, rel_tol=1e-12):
        raise TuningEvidenceError("Performance run FPS does not match its full-process wall time")
    return fps


def _validate_suite(
    assessment: CandidateAssessment,
    *,
    root: Path,
    suite_path: Path,
    workload_id: str,
    variant: str,
    contract_version: int,
    max_relative_spread: float,
    policy: MeasurementPolicy,
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
        expected_parameters = {
            "frames": policy.measured_frames,
            "warmup_frames": policy.warmup_frames,
            "initial_runs": policy.initial_runs,
            "extra_runs_on_spread": policy.extra_runs_on_spread,
            "spread_threshold": policy.spread_threshold,
            "max_relative_spread": policy.max_relative_spread,
            "idle_seconds": policy.idle_seconds,
            "bitrate_validation": policy.bitrate_validation,
        }
        changed = [
            key for key, expected in expected_parameters.items() if parameters.get(key) != expected
        ]
        if changed:
            assessment.reject("Performance suite changed stage parameters: " + ", ".join(changed))
        statistics = suite.get("statistics")
        if not isinstance(statistics, dict):
            raise TuningEvidenceError("Performance suite has no statistics")
        runs = suite.get("runs")
        if not isinstance(runs, list) or not runs:
            raise TuningEvidenceError("Performance suite has no run manifests")
        identities = []
        manifests = []
        values = []
        manifest_paths: set[Path] = set()
        for run_index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                raise TuningEvidenceError("Performance suite run entry is invalid")
            manifest_path = artifact_path(
                root,
                run.get("manifest"),
                label="Performance run",
            )
            if manifest_path in manifest_paths:
                raise TuningEvidenceError("Performance suite repeats a run manifest")
            manifest_paths.add(manifest_path)
            if run.get("index") != run_index or isinstance(run.get("index"), bool):
                raise TuningEvidenceError("Performance suite run indices must be consecutive")
            expected_path = suite_path.parent.resolve() / f"run-{run_index:02d}" / "manifest.json"
            if manifest_path != expected_path:
                raise TuningEvidenceError("Performance run manifest does not belong to this suite")
            if run.get("status") != "valid":
                raise TuningEvidenceError("Performance suite run entry status is not valid")
            if not manifest_path.is_file():
                raise TuningEvidenceError(f"Performance run manifest is missing: {manifest_path}")
            manifest = load_json(manifest_path)
            manifests.append(manifest)
            identities.append(
                validate_run_manifest(
                    manifest,
                    expectation=RunExpectation(
                        product=PRODUCTS[assessment.candidate.implementation],
                        workload_id=workload_id,
                        variant=variant,
                        benchmark_contract_version=contract_version,
                        run_index=run_index,
                        implementation=assessment.candidate.implementation,
                        execution_profile=assessment.candidate.execution_profile(),
                        require_media_validation=True,
                        require_hardware_environment=True,
                    ),
                    checksum_length=64,
                )
            )
            fps = _performance_run_fps(manifest, policy=policy)
            if run.get("end_to_end_fps") != fps:
                raise TuningEvidenceError("Performance suite run FPS differs from its raw manifest")
            values.append(fps)
        computed_statistics = compute_suite_statistics(values)
        changed_statistics = [
            key for key, expected in computed_statistics.items() if statistics.get(key) != expected
        ]
        if changed_statistics:
            assessment.reject(
                "Performance suite statistics differ from raw runs: "
                + ", ".join(changed_statistics)
            )
        assessment.median_fps = computed_statistics["median_fps"]
        assessment.relative_spread = computed_statistics["relative_spread"]
        assert assessment.relative_spread is not None
        if assessment.relative_spread > max_relative_spread:
            assessment.reject(
                "Performance spread exceeds the tuning selection contract "
                f"({assessment.relative_spread:.2%} > {max_relative_spread:.2%})"
            )
        # Match SuiteRunner's decision before the extra runs can change the median.
        expected_runs = policy.initial_runs
        if should_extend_suite(values[: policy.initial_runs], policy.spread_threshold):
            expected_runs += policy.extra_runs_on_spread
        if len(runs) != expected_runs:
            assessment.reject(
                "Performance suite run count does not match its initial spread policy"
            )
        first_identity = identities[0]
        if any(
            identity.implementation_key() != first_identity.implementation_key()
            for identity in identities[1:]
        ):
            assessment.reject("Performance runs changed immutable evidence")
        assessment.identity = first_identity
        if isinstance(assessment.candidate, TasCandidate):
            assessment.median_peak_vram_mib, assessment.median_cpu_cores = resource_medians(
                manifests
            )
    except (ManifestContractError, TuningEvidenceError, ValueError) as exc:
        assessment.reject(str(exc), evidence_complete=False)


def _enforce_shared_contract(assessments: list[CandidateAssessment]) -> None:
    identities = [
        assessment.identity for assessment in assessments if assessment.identity is not None
    ]
    if not identities:
        return
    shared_key = identities[0].shared_model_key()
    for assessment in assessments:
        identity = assessment.identity
        if identity is not None and identity.shared_model_key() != shared_key:
            raise TuningEvidenceError(
                "Tuned evidence changed shared workload, model, encoder, revision, "
                "or CPU/GPU environment contract"
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
                assessment.reject("Candidate changed implementation image or engine")


def _enforce_tuning_session_contract(assessments: list[CandidateAssessment]) -> None:
    identities = [
        assessment.identity for assessment in assessments if assessment.identity is not None
    ]
    if not identities:
        return
    expected_session = identities[0].tuning_session_key()
    if any(identity.tuning_session_key() != expected_session for identity in identities[1:]):
        raise TuningEvidenceError(
            "Tuned evidence changed shared workload, model, encoder, revision, "
            "or CPU/GPU environment contract"
        )
    for implementation in PRODUCTS:
        implementation_identities = [
            assessment.identity
            for assessment in assessments
            if assessment.candidate.implementation == implementation
            and assessment.identity is not None
        ]
        if not implementation_identities:
            continue
        expected_implementation = implementation_identities[0].tuning_implementation_key()
        if any(
            identity.tuning_implementation_key() != expected_implementation
            for identity in implementation_identities[1:]
        ):
            raise TuningEvidenceError(
                f"Tuned evidence changed {implementation} image or engine between search stages"
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
        raise TuningEvidenceError(f"Disqualification evidence is missing: {evidence_path}")
    report = load_json(evidence_path)
    if report.get("document_type") not in {
        "inference-parity",
        "product-output-parity",
    }:
        raise TuningEvidenceError("Candidate disqualification is not a full quality report")
    if report.get("status") != "invalid":
        raise TuningEvidenceError(
            "Candidate disqualification must reference an invalid quality report"
        )
    if report.get("workload_id") != workload_id or report.get("variant") != variant:
        raise TuningEvidenceError("Candidate disqualification changed workload or variant")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise TuningEvidenceError("Candidate disqualification report has no comparisons")
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
        raise TuningEvidenceError("Candidate disqualification execution profile changed")
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
            and identity.implementation_key() != assessment.identity.implementation_key()
        ):
            raise TuningEvidenceError("Candidate disqualification changed implementation evidence")
    assessment.reject("Full winner quality gate failed: " + rejection["reason"])


def _state_entries(
    value: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TuningEvidenceError(f"Adaptive search {label} must be an object array")
    return value


def _assessment_point(assessment: CandidateAssessment) -> CandidatePoint:
    if (
        assessment.median_fps is None
        or assessment.relative_spread is None
        or assessment.suite_path is None
    ):
        raise TuningEvidenceError(
            f"Candidate {assessment.candidate.candidate_id} has incomplete statistics"
        )
    return CandidatePoint(
        candidate=assessment.candidate,
        median_fps=assessment.median_fps,
        relative_spread=assessment.relative_spread,
        suite_path=assessment.suite_path,
        median_peak_vram_mib=assessment.median_peak_vram_mib,
        median_cpu_cores=assessment.median_cpu_cores,
    )


def _validate_state_stage(
    entries: list[dict[str, Any]],
    *,
    implementation: str,
    stage: str,
    contract: TuningContract,
    workload_id: str,
    variant: str,
    contract_version: int,
    sweep_dir: Path,
    root: Path,
    policy: MeasurementPolicy,
) -> list[CandidateAssessment]:
    assessments = []
    seen = set()
    for entry in entries:
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen:
            raise TuningEvidenceError(
                f"Adaptive search {stage} contains an invalid or duplicate candidate"
            )
        seen.add(candidate_id)
        candidate = contract.candidate(candidate_id)
        if candidate.implementation != implementation:
            raise TuningEvidenceError(f"Adaptive search {stage} changed candidate implementation")
        expected_suite = (
            candidate_directory(sweep_dir, candidate) / stage / "performance" / "suite.json"
        )
        suite_value = entry.get("suite")
        if not isinstance(suite_value, str):
            raise TuningEvidenceError(f"Adaptive search {candidate_id} has no suite path")
        if artifact_path(root, suite_value, label=f"{candidate_id} suite") != expected_suite:
            raise TuningEvidenceError(f"Adaptive search {candidate_id} changed its suite path")
        assessment = CandidateAssessment(candidate=candidate)
        _validate_suite(
            assessment,
            root=root,
            suite_path=expected_suite,
            workload_id=workload_id,
            variant=variant,
            contract_version=contract_version,
            max_relative_spread=policy.max_relative_spread,
            policy=policy,
        )
        if assessment.median_fps != entry.get("median_fps"):
            assessment.reject("Adaptive search changed recorded median FPS")
        if assessment.relative_spread != entry.get("relative_spread"):
            assessment.reject("Adaptive search changed recorded relative spread")
        if isinstance(candidate, TasCandidate):
            if (
                entry.get("decode_method") != candidate.decode_method
                or entry.get("writer") != candidate.writer
                or entry.get("cuda_graph") is not True
            ):
                assessment.reject("Finite search changed recorded execution profile")
            if assessment.median_peak_vram_mib != entry.get(
                "median_peak_vram_mib"
            ) or assessment.median_cpu_cores != entry.get("median_cpu_cores"):
                assessment.reject("Finite search changed recorded resource medians")
        assessments.append(assessment)
    return assessments


def _validate_finite_completion(
    *,
    state: dict[str, Any],
    preflight: dict[str, dict[str, Any]],
    reconnaissance: list[CandidateAssessment],
    confirmation: list[CandidateAssessment],
    contract: TuningContract,
    workload_id: str,
    variant: str,
    contract_version: int,
    sweep_dir: Path,
    root: Path,
) -> None:
    """Require every TAS configuration to have measured or hashed OOM evidence."""
    if state.get("completion_reason") != "grid-exhausted":
        raise TuningEvidenceError("Finite TAS search must exhaust the I/O grid")
    if any(
        key in state for key in ("early_stop_after_streams", "cuda_graph_probe", "resource_limit")
    ):
        raise TuningEvidenceError("Finite TAS search cannot contain stream-search decisions")
    points = [_assessment_point(item) for item in reconnaissance]
    measured_ids = [point.candidate.candidate_id for point in points]
    limits = _state_entries(state.get("resource_limits"), label="TAS resource limits")
    limit_ids = []
    for record in limits:
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise TuningEvidenceError("TAS resource limit has no candidate id")
        candidate = contract.candidate(candidate_id)
        if not isinstance(candidate, TasCandidate):
            raise TuningEvidenceError("TAS resource limit changed implementation")
        limit_ids.append(candidate_id)
        suite = candidate_directory(sweep_dir, candidate) / "reconnaissance/performance/suite.json"
        try:
            validate_cuda_oom_record(
                record,
                candidate=candidate,
                policy=contract.search.reconnaissance,
                workload_id=workload_id,
                variant=variant,
                contract_version=contract_version,
                root=root,
                suite_path=suite,
            )
        except ResourceLimitError as exc:
            raise TuningEvidenceError(f"TAS resource-limit evidence is invalid: {exc}") from exc
    actual = measured_ids + limit_ids
    expected_ids = [
        candidate.candidate_id
        for candidate in contract.for_implementation("tas")
        if preflight[candidate.candidate_id]["status"] == "valid"
    ]
    expected_exclusions = [key for key, entry in preflight.items() if entry["status"] == "invalid"]
    if state.get("quality_exclusions") != expected_exclusions:
        raise TuningEvidenceError("Finite TAS search changed quality exclusions")
    if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
        raise TuningEvidenceError("Finite TAS search has incomplete or duplicate grid evidence")
    if measured_ids != [
        candidate_id for candidate_id in expected_ids if candidate_id in measured_ids
    ]:
        raise TuningEvidenceError("Finite TAS reconnaissance changed grid order")
    expected_shortlist = [
        candidate.candidate_id
        for candidate in shortlist(points, size=contract.search.shortlist_size)
    ]
    if state.get("shortlist") != expected_shortlist:
        raise TuningEvidenceError("Finite TAS search changed its shortlist")
    if {item.candidate.candidate_id for item in confirmation} != set(expected_shortlist):
        raise TuningEvidenceError(
            "Finite TAS search must confirm exactly its best three candidates"
        )


def _validate_completion(
    *,
    implementation: str,
    state: dict[str, Any],
    reconnaissance: list[CandidateAssessment],
    confirmation: list[CandidateAssessment],
    contract: TuningContract,
    workload_id: str,
    variant: str,
    contract_version: int,
    sweep_dir: Path,
    root: Path,
    preflight: dict[str, dict[str, Any]],
) -> None:
    if implementation == "tas":
        _validate_finite_completion(
            state=state,
            preflight=preflight,
            reconnaissance=reconnaissance,
            confirmation=confirmation,
            contract=contract,
            workload_id=workload_id,
            variant=variant,
            contract_version=contract_version,
            sweep_dir=sweep_dir,
            root=root,
        )
        return
    reconnaissance_points = [_assessment_point(item) for item in reconnaissance]
    streams = [stream_count(point.candidate) for point in reconnaissance_points]
    if streams != sorted(set(streams)):
        raise TuningEvidenceError(
            f"Adaptive search {implementation} reconnaissance is not stream-ordered"
        )
    reason = state.get("completion_reason")
    early_stop_after = state.get("early_stop_after_streams")
    resource_limit = state.get("resource_limit")
    full_range = list(contract.search.stream_range)
    if reason in {"range-exhausted", "sentinel-recovery-range-exhausted"}:
        if resource_limit is not None:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} recorded an unexpected resource limit"
            )
        if streams != full_range:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} did not exhaust the stream range"
            )
        if reason == "range-exhausted" and early_stop_after is not None:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} recorded an unexpected early stop"
            )
        if reason == "sentinel-recovery-range-exhausted" and not isinstance(
            early_stop_after,
            int,
        ):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} has no recovered early-stop point"
            )
        if upper_boundary_unresolved(
            reconnaissance_points,
            relative_margin=contract.search.decline_margin,
        ):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} ended at an increasing upper boundary"
            )
    elif reason == "decline-confirmed":
        if resource_limit is not None:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} recorded an unexpected resource limit"
            )
        if not isinstance(early_stop_after, int):
            raise TuningEvidenceError(f"Adaptive search {implementation} has no early-stop point")
        expected = list(range(contract.search.minimum_streams, early_stop_after + 1))
        expected.append(contract.search.sentinel_streams)
        if streams != expected:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} changed the decline-confirmed range"
            )
        regular = [
            point
            for point in reconnaissance_points
            if stream_count(point.candidate) != contract.search.sentinel_streams
        ]
        sentinel = reconnaissance_points[-1]
        if not has_confirmed_decline(
            regular,
            relative_margin=contract.search.decline_margin,
            patience=contract.search.decline_patience,
        ):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} stopped without a confirmed decline"
            )
        if sentinel_recovers(
            regular,
            sentinel,
            relative_margin=contract.search.decline_margin,
        ):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} ignored a recovering sentinel"
            )
    elif reason == "resource-ceiling":
        if not isinstance(resource_limit, dict):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} has no resource-limit evidence"
            )
        candidate_id = resource_limit.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} resource limit has no candidate"
            )
        candidate = contract.candidate(candidate_id)
        if candidate.implementation != implementation or candidate.cuda_graph:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} changed its resource-limit candidate"
            )
        if resource_limit.get("num_streams") != stream_count(candidate):
            raise TuningEvidenceError(
                f"Adaptive search {implementation} changed its resource-limit stream count"
            )
        expected_suite = (
            candidate_directory(sweep_dir, candidate)
            / "reconnaissance"
            / "performance"
            / "suite.json"
        )
        try:
            validate_cuda_oom_record(
                resource_limit,
                candidate=candidate,
                policy=contract.search.reconnaissance,
                workload_id=workload_id,
                variant=variant,
                contract_version=contract_version,
                root=root,
                suite_path=expected_suite,
            )
        except ResourceLimitError as exc:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} resource-limit evidence is invalid: {exc}"
            ) from exc
        if not streams or stream_count(candidate) <= streams[-1]:
            raise TuningEvidenceError(
                f"Adaptive search {implementation} resource ceiling is not above valid points"
            )
        if early_stop_after is None:
            expected = list(range(contract.search.minimum_streams, stream_count(candidate)))
            if streams != expected:
                raise TuningEvidenceError(
                    f"Adaptive search {implementation} skipped points before its resource ceiling"
                )
        else:
            if (
                not isinstance(early_stop_after, int)
                or stream_count(candidate) != contract.search.sentinel_streams
            ):
                raise TuningEvidenceError(
                    f"Adaptive search {implementation} has an invalid OOM sentinel stop"
                )
            expected = list(range(contract.search.minimum_streams, early_stop_after + 1))
            if streams != expected or not has_confirmed_decline(
                reconnaissance_points,
                relative_margin=contract.search.decline_margin,
                patience=contract.search.decline_patience,
            ):
                raise TuningEvidenceError(
                    f"Adaptive search {implementation} has an unproven pre-sentinel decline"
                )
    else:
        raise TuningEvidenceError(
            f"Adaptive search {implementation} has an invalid completion reason"
        )

    expected_shortlist = [
        candidate.candidate_id
        for candidate in shortlist(
            reconnaissance_points,
            size=contract.search.shortlist_size,
        )
    ]
    if state.get("shortlist") != expected_shortlist:
        raise TuningEvidenceError(f"Adaptive search {implementation} changed its shortlist")
    confirmation_by_id = {
        assessment.candidate.candidate_id: assessment for assessment in confirmation
    }
    if not set(expected_shortlist).issubset(confirmation_by_id):
        raise TuningEvidenceError(
            f"Adaptive search {implementation} has incomplete confirmation evidence"
        )
    provisional = select_peak_equivalent(
        [
            _assessment_point(confirmation_by_id[candidate_id])
            for candidate_id in expected_shortlist
        ],
        equivalence_margin=contract.selection.equivalence_margin,
    )
    if provisional is None:
        raise TuningEvidenceError(f"Adaptive search {implementation} has no provisional winner")
    graph_probe = state.get("cuda_graph_probe")
    profile = contract.implementation(provisional.candidate.implementation)
    expected_graph = (
        contract.make_candidate(
            provisional.candidate.implementation,
            stream_count(provisional.candidate),
            cuda_graph=True,
        ).candidate_id
        if isinstance(profile, ImplementationProfile)
        and profile.probe_cuda_graph
        and isinstance(provisional.candidate, TunedCandidate)
        else None
    )
    if graph_probe != expected_graph:
        raise TuningEvidenceError(f"Adaptive search {implementation} changed its CUDA Graph probe")
    expected_confirmation = set(expected_shortlist)
    if expected_graph is not None:
        expected_confirmation.add(expected_graph)
    if set(confirmation_by_id) != expected_confirmation:
        raise TuningEvidenceError(
            f"Adaptive search {implementation} has unexpected confirmation candidates"
        )


def _load_adaptive_assessments(
    *,
    contract: TuningContract,
    workload: dict[str, Any],
    variant: str,
    sweep_dir: Path,
    root: Path,
) -> tuple[list[CandidateAssessment], list[CandidateAssessment], dict[str, Any]]:
    state_path = sweep_dir / "search-state.json"
    if not state_path.is_file():
        raise TuningEvidenceError(f"Adaptive search state is missing: {state_path}")
    state = load_json(state_path)
    contract_version = int(workload["benchmark"]["contract_version"])
    expected = {
        "schema_version": 3,
        "document_type": "adaptive-tuning-search",
        "status": "complete",
        "workload_id": workload["id"],
        "variant": variant,
        "benchmark_contract_version": contract_version,
        "search_policy": contract.search.as_dict(),
        "selection_policy": contract.selection.as_dict(),
    }
    changed = [key for key, value in expected.items() if state.get(key) != value]
    if changed:
        raise TuningEvidenceError(
            "Adaptive search state changed its contract: " + ", ".join(changed)
        )
    implementations = state.get("implementations")
    if not isinstance(implementations, dict) or set(implementations) != set(PRODUCTS):
        raise TuningEvidenceError("Adaptive search state must contain vstrt and tas")
    preflight_record = state.get("tas_preflight")
    preflight_path = sweep_dir / "tas-preflight" / "preflight.json"
    if (
        not isinstance(preflight_record, dict)
        or artifact_path(root, preflight_record.get("path"), label="TAS preflight")
        != preflight_path
    ):
        raise TuningEvidenceError("Adaptive search changed its TAS preflight path")
    _verified_quality_report(root, preflight_record)
    preflight = load_tas_preflight(
        preflight_path,
        root=root,
        contract=contract,
        workload=workload,
        variant=variant,
    )

    reconnaissance_all = []
    confirmation_all = []
    for implementation in PRODUCTS:
        implementation_state = implementations[implementation]
        if not isinstance(implementation_state, dict):
            raise TuningEvidenceError(f"Adaptive search {implementation} state must be an object")
        reconnaissance = _validate_state_stage(
            _state_entries(
                implementation_state.get("reconnaissance"),
                label=f"{implementation} reconnaissance",
            ),
            implementation=implementation,
            stage="reconnaissance",
            contract=contract,
            workload_id=workload["id"],
            variant=variant,
            contract_version=contract_version,
            sweep_dir=sweep_dir,
            root=root,
            policy=contract.search.reconnaissance,
        )
        confirmation = _validate_state_stage(
            _state_entries(
                implementation_state.get("confirmation"),
                label=f"{implementation} confirmation",
            ),
            implementation=implementation,
            stage="confirmation",
            contract=contract,
            workload_id=workload["id"],
            variant=variant,
            contract_version=contract_version,
            sweep_dir=sweep_dir,
            root=root,
            policy=contract.search.confirmation,
        )
        _validate_completion(
            implementation=implementation,
            state=implementation_state,
            reconnaissance=reconnaissance,
            confirmation=confirmation,
            contract=contract,
            workload_id=workload["id"],
            variant=variant,
            contract_version=contract_version,
            sweep_dir=sweep_dir,
            root=root,
            preflight=preflight,
        )
        reconnaissance_all.extend(reconnaissance)
        confirmation_all.extend(confirmation)
    for assessment in [*reconnaissance_all, *confirmation_all]:
        if isinstance(assessment.candidate, TasCandidate) and assessment.identity is not None:
            record = preflight[assessment.candidate.candidate_id]
            product = _verified_quality_report(root, record["quality"]["product_output"])
            comparison = next(
                row for row in product["comparisons"] if row["implementation"] == PRODUCTS["tas"]
            )
            identity = _preflight_product_identity(
                comparison, assessment.candidate, root, workload, variant
            )
            if (
                identity.tuning_implementation_key()
                != assessment.identity.tuning_implementation_key()
            ):
                raise TuningEvidenceError(
                    "TAS preflight changed image, engine, or session evidence"
                )
    return reconnaissance_all, confirmation_all, state


def _winner(
    assessments: list[CandidateAssessment],
    implementation: str,
    *,
    equivalence_margin: float,
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

    maximum = max(
        assessment.median_fps for assessment in eligible if assessment.median_fps is not None
    )
    equivalent = [
        assessment
        for assessment in eligible
        if assessment.median_fps is not None
        and assessment.median_fps >= maximum * (1 - equivalence_margin)
    ]
    return min(
        equivalent,
        key=lambda assessment: resource_order(_assessment_point(assessment)),
    )


def rank_tuned_candidates(
    *,
    contract: TuningContract,
    workload: dict[str, Any],
    variant: str,
    sweep_dir: Path,
    root: Path,
    disqualifications: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate adaptive search evidence and select one winner per competitor."""
    root = root.resolve()
    sweep_dir = sweep_dir.resolve()
    contract_version = int(workload["benchmark"]["contract_version"])
    reconnaissance, assessments, search_state = _load_adaptive_assessments(
        contract=contract,
        workload=workload,
        variant=variant,
        sweep_dir=sweep_dir,
        root=root,
    )
    _enforce_shared_contract(reconnaissance)
    _enforce_shared_contract(assessments)
    _enforce_tuning_session_contract([*reconnaissance, *assessments])
    search_errors = [
        f"{assessment.candidate.candidate_id}: {error}"
        for assessment in [*reconnaissance, *assessments]
        for error in assessment.errors
    ]
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
        implementation: _winner(
            assessments,
            implementation,
            equivalence_margin=contract.selection.equivalence_margin,
        )
        for implementation in PRODUCTS
    }
    incomplete = [
        assessment.candidate.candidate_id
        for assessment in assessments
        if not assessment.evidence_complete
    ]
    errors = []
    if incomplete:
        errors.append(
            "Complete confirmation evidence is missing for shortlisted candidates: "
            + ", ".join(incomplete)
        )
    errors.extend(search_errors)
    for implementation, winner in winners.items():
        if winner is None:
            errors.append(f"No eligible {implementation} candidate remains")

    first_identity = next(
        (assessment.identity for assessment in assessments if assessment.identity is not None),
        None,
    )
    hardware_environment = (
        first_identity.environment
        if first_identity is not None and first_identity.environment is not None
        else {}
    )
    report = {
        "schema_version": 3,
        "document_type": "tuned-candidate-selection",
        "status": "valid" if not errors else "invalid",
        "publishable": False,
        "scope": "tuning-selection",
        "workload_id": workload["id"],
        "variant": variant,
        "benchmark_contract_version": contract_version,
        "selection_policy": contract.selection.as_dict(),
        "tas_preflight": search_state["tas_preflight"],
        "search": {
            "path": (sweep_dir / "search-state.json").relative_to(root).as_posix(),
            "completion": {
                implementation: search_state["implementations"][implementation]["completion_reason"]
                for implementation in PRODUCTS
            },
            "resource_limits": {
                implementation: search_state["implementations"][implementation].get(
                    "resource_limits" if implementation == "tas" else "resource_limit"
                )
                for implementation in PRODUCTS
            },
        },
        "project_profile": contract.project_profile.as_dict(),
        "disqualifications": disqualifications,
        "environment": {
            "repository_revision": (
                first_identity.repository_revision if first_identity is not None else None
            ),
            **hardware_environment,
        },
        "reconnaissance": [assessment.as_dict() for assessment in reconnaissance],
        "candidates": [assessment.as_dict() for assessment in assessments],
        "winners": {
            implementation: (
                {
                    "candidate_id": winner.candidate.candidate_id,
                    "median_fps": winner.median_fps,
                    "relative_spread": winner.relative_spread,
                    **(
                        {
                            "median_peak_vram_mib": winner.median_peak_vram_mib,
                            "median_cpu_cores": winner.median_cpu_cores,
                        }
                        if isinstance(winner.candidate, TasCandidate)
                        else {}
                    ),
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
