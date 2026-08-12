"""Run the declared tuned sweep and finalize its selected candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

from benchmarks.scripts.contracts.manifest import execution_profile
from benchmarks.scripts.tuning.adaptive import (
    CandidatePoint,
    has_confirmed_decline,
    select_peak_equivalent,
    sentinel_recovers,
    shortlist,
    upper_boundary_unresolved,
)
from benchmarks.scripts.tuning.contract import (
    Implementation,
    MeasurementPolicy,
    TunedCandidate,
    TuningContract,
    TuningContractError,
    load_tuning_contract,
)
from benchmarks.scripts.tuning.rank import (
    PRODUCTS,
    TuningEvidenceError,
    candidate_directory,
    load_disqualifications,
    rank_tuned_candidates,
)
from benchmarks.scripts.tuning.resource_limit import (
    ResourceLimitEvidence,
    detect_cuda_oom,
)
from benchmarks.scripts.workloads.manifest import load_manifest


class TuningWorkflowError(RuntimeError):
    """Raised when a tuned workflow cannot preserve its evidence contract."""


def _progress(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningWorkflowError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TuningWorkflowError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkflowPaths:
    """Repository and artifact paths shared by tuning commands."""

    root: Path
    benchmarks_dir: Path
    manifest: Path
    sweep_dir: Path

    @classmethod
    def resolve(
        cls,
        *,
        root: Path,
        benchmarks_dir: Path,
        manifest: Path,
        sweep_dir: Path,
    ) -> WorkflowPaths:
        resolved_root = root.resolve()

        def under_root(path: Path, *, label: str) -> Path:
            resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
            if resolved != resolved_root and resolved_root not in resolved.parents:
                raise TuningWorkflowError(f"{label} escapes the repository root")
            return resolved

        return cls(
            root=resolved_root,
            benchmarks_dir=under_root(
                benchmarks_dir,
                label="Benchmarks directory",
            ),
            manifest=under_root(manifest, label="Workload manifest"),
            sweep_dir=under_root(sweep_dir, label="Sweep directory"),
        )

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise TuningWorkflowError(f"Artifact is outside the repository root: {path}") from exc


@dataclass(frozen=True)
class MakeRunner:
    """Invoke benchmark Make targets without shell interpolation."""

    paths: WorkflowPaths
    executable: str = "make"

    def run(
        self,
        target: str,
        variables: dict[str, str],
        *,
        accepted_artifact: Path | None = None,
    ) -> int:
        command = [
            self.executable,
            "-C",
            str(self.paths.benchmarks_dir),
            target,
            *[f"{key}={value}" for key, value in variables.items()],
        ]
        result = subprocess.run(command, cwd=self.paths.root, check=False)
        if result.returncode != 0 and (
            accepted_artifact is None or not accepted_artifact.is_file()
        ):
            raise TuningWorkflowError(
                f"Make target {target!r} failed with code {result.returncode}"
            )
        return result.returncode


def _require_clean_destination(
    directory: Path,
    *,
    marker: Path,
    resume: bool,
) -> bool:
    """Return whether work is required, rejecting ambiguous partial output."""
    if marker.is_file():
        if not resume:
            raise TuningWorkflowError(
                f"Evidence already exists; use --resume or choose another path: {marker}"
            )
        return False
    if directory.exists() and any(directory.iterdir()):
        raise TuningWorkflowError(f"Partial evidence must be removed before retrying: {directory}")
    return True


def _base_make_variables(
    paths: WorkflowPaths,
    *,
    variant: str,
    engine: Path,
    vsgan_engine: Path,
    gpu_id: int,
) -> dict[str, str]:
    return {
        "MANIFEST": paths.relative(paths.manifest),
        "VARIANT": variant,
        "ENGINE": paths.relative(engine),
        "VSGAN_ENGINE": paths.relative(vsgan_engine),
        "GPU_ID": str(gpu_id),
        "EXECUTION_PROFILE": "tuned",
    }


def _candidate_variables(
    candidate: TunedCandidate,
    base: dict[str, str],
) -> dict[str, str]:
    values = dict(base)
    values["VSTRT_ARGS" if candidate.implementation == "vstrt" else "VSGAN_ARGS"] = (
        candidate.runner_arguments()
    )
    return values


def _write_selection(
    *,
    contract: TuningContract,
    workload: dict[str, Any],
    variant: str,
    paths: WorkflowPaths,
    disqualifications_path: Path,
) -> dict[str, Any]:
    disqualifications = load_disqualifications(
        disqualifications_path,
        contract=contract,
    )
    report = rank_tuned_candidates(
        contract=contract,
        workload=workload,
        variant=variant,
        sweep_dir=paths.sweep_dir,
        root=paths.root,
        disqualifications=disqualifications,
    )
    _write_json(paths.sweep_dir / "selection.json", report)
    return report


def _validate_search_suite_contract(
    *,
    suite: dict[str, Any],
    candidate: TunedCandidate,
    policy: MeasurementPolicy,
    suite_path: Path,
) -> None:
    parameters = suite.get("parameters")
    if not isinstance(parameters, dict):
        raise TuningWorkflowError(f"Search suite is incomplete: {suite_path}")
    if execution_profile(parameters) != candidate.execution_profile():
        raise TuningWorkflowError(
            f"Search suite changed execution profile for {candidate.candidate_id}"
        )
    expected = {
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
        key for key, expected_value in expected.items() if parameters.get(key) != expected_value
    ]
    if changed:
        raise TuningWorkflowError(
            f"Search suite changed {candidate.candidate_id} stage parameters: " + ", ".join(changed)
        )


def _search_point(
    *,
    candidate: TunedCandidate,
    suite_path: Path,
    policy: MeasurementPolicy,
    paths: WorkflowPaths,
) -> CandidatePoint:
    suite = _load_json(suite_path)
    _validate_search_suite_contract(
        suite=suite,
        candidate=candidate,
        policy=policy,
        suite_path=suite_path,
    )
    if suite.get("status") != "valid":
        raise TuningWorkflowError(
            f"Search suite is not valid for {candidate.candidate_id}: {suite_path}"
        )
    statistics = suite.get("statistics")
    if not isinstance(statistics, dict):
        raise TuningWorkflowError(f"Search suite has no statistics: {suite_path}")
    median_fps = statistics.get("median_fps")
    relative_spread = statistics.get("relative_spread")
    if not isinstance(median_fps, (int, float)) or isinstance(median_fps, bool) or median_fps <= 0:
        raise TuningWorkflowError(f"Search suite has no positive median FPS: {suite_path}")
    if (
        not isinstance(relative_spread, (int, float))
        or isinstance(relative_spread, bool)
        or relative_spread < 0
    ):
        raise TuningWorkflowError(f"Search suite has no valid relative spread: {suite_path}")
    return CandidatePoint(
        candidate=candidate,
        median_fps=float(median_fps),
        relative_spread=float(relative_spread),
        suite_path=paths.relative(suite_path),
    )


@overload
def _measure_search_candidate(
    *,
    candidate: TunedCandidate,
    stage: str,
    policy: MeasurementPolicy,
    base: dict[str, str],
    paths: WorkflowPaths,
    runner: MakeRunner,
    resume: bool,
    allow_resource_limit: Literal[False] = False,
) -> CandidatePoint: ...


@overload
def _measure_search_candidate(
    *,
    candidate: TunedCandidate,
    stage: str,
    policy: MeasurementPolicy,
    base: dict[str, str],
    paths: WorkflowPaths,
    runner: MakeRunner,
    resume: bool,
    allow_resource_limit: Literal[True],
) -> CandidatePoint | ResourceLimitEvidence: ...


def _measure_search_candidate(
    *,
    candidate: TunedCandidate,
    stage: str,
    policy: MeasurementPolicy,
    base: dict[str, str],
    paths: WorkflowPaths,
    runner: MakeRunner,
    resume: bool,
    allow_resource_limit: bool = False,
) -> CandidatePoint | ResourceLimitEvidence:
    performance_dir = candidate_directory(paths.sweep_dir, candidate) / stage / "performance"
    suite_path = performance_dir / "suite.json"
    required = _require_clean_destination(
        performance_dir,
        marker=suite_path,
        resume=resume,
    )
    _progress(
        f"[tuned {stage}] {candidate.candidate_id}: "
        + ("performance suite" if required else "SKIP performance suite")
    )
    if required:
        output_variable = (
            "VSTRT_OUTPUT_DIR" if candidate.implementation == "vstrt" else "VSGAN_OUTPUT_DIR"
        )
        runner.run(
            f"run-{candidate.implementation}",
            {
                **_candidate_variables(candidate, base),
                output_variable: paths.relative(performance_dir),
                "ARGS": policy.runner_arguments(),
            },
            accepted_artifact=suite_path,
        )
    try:
        return _search_point(
            candidate=candidate,
            suite_path=suite_path,
            policy=policy,
            paths=paths,
        )
    except TuningWorkflowError:
        suite = _load_json(suite_path)
        _validate_search_suite_contract(
            suite=suite,
            candidate=candidate,
            policy=policy,
            suite_path=suite_path,
        )
        evidence = None
        if allow_resource_limit and suite.get("status") == "invalid":
            evidence = detect_cuda_oom(root=paths.root, suite_path=suite_path)
        if evidence is None:
            raise
        _progress(
            f"[tuned {stage}] {candidate.candidate_id}: CUDA OOM recorded as resource ceiling"
        )
        return evidence


def _resource_limit_record(
    candidate: TunedCandidate,
    evidence: ResourceLimitEvidence,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "num_streams": candidate.num_streams,
        **evidence.as_dict(),
    }


def _run_reconnaissance(
    *,
    implementation: Implementation,
    contract: TuningContract,
    base: dict[str, str],
    paths: WorkflowPaths,
    runner: MakeRunner,
    resume: bool,
) -> tuple[list[CandidatePoint], str, int | None, dict[str, Any] | None]:
    policy = contract.search.reconnaissance
    points: list[CandidatePoint] = []
    early_stop_after = None
    for streams in contract.search.stream_range:
        candidate = contract.make_candidate(implementation, streams)
        measured = _measure_search_candidate(
            candidate=candidate,
            stage="reconnaissance",
            policy=policy,
            base=base,
            paths=paths,
            runner=runner,
            resume=resume,
            allow_resource_limit=True,
        )
        if isinstance(measured, ResourceLimitEvidence):
            if not points:
                raise TuningWorkflowError(
                    f"{implementation} cannot execute its minimum stream candidate"
                )
            return (
                points,
                "resource-ceiling",
                None,
                _resource_limit_record(candidate, measured),
            )
        points.append(measured)
        if streams == contract.search.maximum_streams:
            break
        if has_confirmed_decline(
            points,
            relative_margin=contract.search.decline_margin,
            patience=contract.search.decline_patience,
        ):
            early_stop_after = streams
            break

    if early_stop_after is None:
        if upper_boundary_unresolved(
            points,
            relative_margin=contract.search.decline_margin,
        ):
            raise TuningWorkflowError(
                f"{implementation} throughput is still increasing at "
                f"num_streams={contract.search.maximum_streams}; expand the "
                "tuning contract range"
            )
        return points, "range-exhausted", None, None

    sentinel_candidate = contract.make_candidate(
        implementation,
        contract.search.sentinel_streams,
    )
    sentinel = _measure_search_candidate(
        candidate=sentinel_candidate,
        stage="reconnaissance",
        policy=policy,
        base=base,
        paths=paths,
        runner=runner,
        resume=resume,
        allow_resource_limit=True,
    )
    if isinstance(sentinel, ResourceLimitEvidence):
        return (
            points,
            "resource-ceiling",
            early_stop_after,
            _resource_limit_record(sentinel_candidate, sentinel),
        )
    if sentinel_recovers(
        points,
        sentinel,
        relative_margin=contract.search.decline_margin,
    ):
        measured_streams = {point.candidate.num_streams for point in points}
        measured_streams.add(sentinel.candidate.num_streams)
        for streams in contract.search.stream_range:
            if streams in measured_streams:
                continue
            points.append(
                _measure_search_candidate(
                    candidate=contract.make_candidate(implementation, streams),
                    stage="reconnaissance",
                    policy=policy,
                    base=base,
                    paths=paths,
                    runner=runner,
                    resume=resume,
                )
            )
        completed = sorted(
            points + [sentinel],
            key=lambda point: point.candidate.num_streams,
        )
        if upper_boundary_unresolved(
            completed,
            relative_margin=contract.search.decline_margin,
        ):
            raise TuningWorkflowError(
                f"{implementation} throughput recovered and is still increasing at "
                f"num_streams={contract.search.maximum_streams}; expand the "
                "tuning contract range"
            )
        return (
            completed,
            "sentinel-recovery-range-exhausted",
            early_stop_after,
            None,
        )
    return (
        sorted(points + [sentinel], key=lambda point: point.candidate.num_streams),
        "decline-confirmed",
        early_stop_after,
        None,
    )


def _run_confirmation(
    *,
    implementation: Implementation,
    reconnaissance: list[CandidatePoint],
    contract: TuningContract,
    base: dict[str, str],
    paths: WorkflowPaths,
    runner: MakeRunner,
    resume: bool,
) -> tuple[list[CandidatePoint], TunedCandidate | None]:
    selected = shortlist(
        reconnaissance,
        size=contract.search.shortlist_size,
    )
    confirmed = [
        _measure_search_candidate(
            candidate=candidate,
            stage="confirmation",
            policy=contract.search.confirmation,
            base=base,
            paths=paths,
            runner=runner,
            resume=resume,
        )
        for candidate in selected
    ]
    provisional = select_peak_equivalent(
        confirmed,
        equivalence_margin=contract.selection.equivalence_margin,
    )
    if provisional is None:
        raise TuningWorkflowError(f"No confirmed {implementation} candidate remains")
    graph_candidate = None
    if contract.implementation(implementation).probe_cuda_graph:
        graph_candidate = contract.make_candidate(
            implementation,
            provisional.candidate.num_streams,
            cuda_graph=True,
        )
        confirmed.append(
            _measure_search_candidate(
                candidate=graph_candidate,
                stage="confirmation",
                policy=contract.search.confirmation,
                base=base,
                paths=paths,
                runner=runner,
                resume=resume,
            )
        )
    return confirmed, graph_candidate


def _write_or_verify_search_state(
    path: Path,
    value: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if path.is_file():
        if not resume:
            raise TuningWorkflowError(f"Search state already exists: {path}")
        if _load_json(path) != value:
            raise TuningWorkflowError(
                "Adaptive search decisions changed while resuming; start a new sweep"
            )
        return
    _write_json(path, value)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run adaptive reconnaissance and confirm only the strongest candidates."""
    paths = _workflow_paths(args)
    contract = load_tuning_contract(
        _required_file(paths, Path(args.contract), label="Tuning contract")
    )
    workload = load_manifest(paths.manifest)
    engine = _required_file(paths, Path(args.engine), label="Engine")
    vsgan_engine = _required_file(
        paths,
        Path(args.vsgan_engine),
        label="VSGAN engine",
    )
    runner = MakeRunner(paths, executable=args.make)
    if paths.sweep_dir.exists() and any(paths.sweep_dir.iterdir()) and not args.resume:
        raise TuningWorkflowError(f"Sweep directory is not empty; use --resume: {paths.sweep_dir}")
    paths.sweep_dir.mkdir(parents=True, exist_ok=True)
    base = _base_make_variables(
        paths,
        variant=args.variant,
        engine=engine,
        vsgan_engine=vsgan_engine,
        gpu_id=args.gpu_id,
    )

    implementation_states = {}
    for implementation in PRODUCTS:
        implementation_name = cast(Implementation, implementation)
        _progress(f"[tuned search] {implementation}: reconnaissance")
        (
            reconnaissance,
            completion_reason,
            early_stop_after,
            resource_limit,
        ) = _run_reconnaissance(
            implementation=implementation_name,
            contract=contract,
            base=base,
            paths=paths,
            runner=runner,
            resume=args.resume,
        )
        _progress(f"[tuned search] {implementation}: confirmation")
        confirmation, graph_candidate = _run_confirmation(
            implementation=implementation_name,
            reconnaissance=reconnaissance,
            contract=contract,
            base=base,
            paths=paths,
            runner=runner,
            resume=args.resume,
        )
        implementation_states[implementation] = {
            "completion_reason": completion_reason,
            "early_stop_after_streams": early_stop_after,
            "resource_limit": resource_limit,
            "reconnaissance": [point.as_dict() for point in reconnaissance],
            "shortlist": [
                candidate.candidate_id
                for candidate in shortlist(
                    reconnaissance,
                    size=contract.search.shortlist_size,
                )
            ],
            "confirmation": [point.as_dict() for point in confirmation],
            "cuda_graph_probe": (
                graph_candidate.candidate_id if graph_candidate is not None else None
            ),
        }

    state = {
        "schema_version": 2,
        "document_type": "adaptive-tuning-search",
        "status": "complete",
        "workload_id": workload["id"],
        "variant": args.variant,
        "benchmark_contract_version": workload["benchmark"]["contract_version"],
        "contract": {
            "path": paths.relative(_required_file(paths, Path(args.contract), label="contract")),
            "sha256": _sha256(_under_root(paths, Path(args.contract), label="contract")),
            "schema_version": contract.schema_version,
        },
        "workload_sha256": _sha256(paths.manifest),
        "search_policy": contract.search.as_dict(),
        "selection_policy": contract.selection.as_dict(),
        "implementations": implementation_states,
    }
    _write_or_verify_search_state(
        paths.sweep_dir / "search-state.json",
        state,
        resume=args.resume,
    )

    report = _write_selection(
        contract=contract,
        workload=workload,
        variant=args.variant,
        paths=paths,
        disqualifications_path=paths.sweep_dir / "disqualifications.json",
    )
    if report["status"] != "valid":
        raise TuningWorkflowError(
            f"Tuned selection is invalid; inspect {paths.sweep_dir / 'selection.json'}"
        )
    _progress(
        "[tuned sweep] Selected "
        + ", ".join(
            f"{implementation}={winner['candidate_id']}"
            for implementation, winner in sorted(report["winners"].items())
        )
    )
    return report


def _selected_candidates(
    selection: dict[str, Any],
    contract: TuningContract,
) -> dict[str, TunedCandidate]:
    if selection.get("status") != "valid":
        raise TuningWorkflowError("Tuned candidate selection is not valid")
    winners = selection.get("winners")
    if not isinstance(winners, dict):
        raise TuningWorkflowError("Tuned candidate selection has no winners")
    result = {}
    for implementation in PRODUCTS:
        winner = winners.get(implementation)
        candidate_id = winner.get("candidate_id") if isinstance(winner, dict) else None
        if not isinstance(candidate_id, str):
            raise TuningWorkflowError(f"Tuned candidate selection has no {implementation} winner")
        result[implementation] = contract.candidate(candidate_id)
    return result


def _winner_signature(winners: dict[str, TunedCandidate]) -> str:
    return "__".join(winners[implementation].candidate_id for implementation in sorted(winners))


def _failed_implementations(report_path: Path) -> dict[str, str]:
    report = _load_json(report_path)
    failures = {}
    for comparison in report.get("comparisons", []):
        if not isinstance(comparison, dict) or comparison.get("status") == "valid":
            continue
        product = comparison.get("implementation")
        implementation = next(
            (name for name, expected in PRODUCTS.items() if product == expected),
            None,
        )
        if implementation is None:
            continue
        errors = comparison.get("errors")
        reason = "; ".join(str(error) for error in errors) if errors else "invalid"
        failures[implementation] = reason
    return failures


def _common_product_output_failure(report_path: Path) -> str | None:
    """Identify one byte-identical invalid output shared by both competitors."""
    report = _load_json(report_path)
    comparisons = [
        comparison for comparison in report.get("comparisons", []) if isinstance(comparison, dict)
    ]
    if (
        len(comparisons) != len(PRODUCTS)
        or {comparison.get("implementation") for comparison in comparisons}
        != set(PRODUCTS.values())
        or any(comparison.get("status") == "valid" for comparison in comparisons)
    ):
        return None
    output_hashes: set[str] = set()
    for comparison in comparisons:
        output_hash = comparison.get("output_sha256")
        if not isinstance(output_hash, str) or not output_hash:
            return None
        output_hashes.add(output_hash)
    if len(output_hashes) != 1:
        return None
    output_sha256 = next(iter(output_hashes))
    reasons = sorted(
        {str(error) for comparison in comparisons for error in comparison.get("errors", [])}
    )
    detail = "; ".join(reasons) if reasons else "product-output gate failed"
    return (
        "Both external implementations produced the same invalid MP4 "
        f"({output_sha256[:12]}): {detail}. This is a common product-path "
        "failure, not evidence against either scheduling candidate"
    )


def _failed_inference_implementations(report_path: Path) -> dict[str, str]:
    """Return model-output failures; shared-input or identity failures are fatal."""
    report = _load_json(report_path)
    for comparison in report.get("comparisons", []):
        if not isinstance(comparison, dict) or comparison.get("status") == "valid":
            continue
        errors = [str(error) for error in comparison.get("errors", [])]
        infrastructure_errors = [error for error in errors if not error.startswith("output frame ")]
        if infrastructure_errors:
            implementation = comparison.get("implementation", "unknown implementation")
            raise TuningWorkflowError(
                f"Shared-input inference evidence failed for {implementation}: "
                + "; ".join(infrastructure_errors)
            )
    return _failed_implementations(report_path)


def _record_disqualifications(
    path: Path,
    *,
    winners: dict[str, TunedCandidate],
    failures: dict[str, str],
    evidence: Path,
    paths: WorkflowPaths,
) -> None:
    value = _load_json(path) if path.is_file() else {"schema_version": 1, "entries": []}
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise TuningWorkflowError("Disqualification entries must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        if isinstance(candidate_id, str):
            by_id[candidate_id] = entry
    for implementation, reason in failures.items():
        candidate = winners[implementation]
        by_id[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "reason": reason,
            "evidence": paths.relative(evidence),
        }
    value["entries"] = [by_id[candidate_id] for candidate_id in sorted(by_id)]
    _write_json(path, value)


def run_winner_quality(args: argparse.Namespace) -> dict[str, Any]:
    """Run full quality gates, promoting the next candidate after a failure."""
    paths = _workflow_paths(args)
    contract = load_tuning_contract(
        _required_file(paths, Path(args.contract), label="Tuning contract")
    )
    workload = load_manifest(paths.manifest)
    engine = _required_file(paths, Path(args.engine), label="Engine")
    vsgan_engine = _required_file(
        paths,
        Path(args.vsgan_engine),
        label="VSGAN engine",
    )
    runner = MakeRunner(paths, executable=args.make)
    disqualifications_path = paths.sweep_dir / "disqualifications.json"

    for attempt in range(1, len(contract.candidates) + 1):
        selection = _write_selection(
            contract=contract,
            workload=workload,
            variant=args.variant,
            paths=paths,
            disqualifications_path=disqualifications_path,
        )
        winners = _selected_candidates(selection, contract)
        signature = _winner_signature(winners)
        _progress(
            f"[tuned quality attempt {attempt}] "
            + ", ".join(
                f"{implementation}={candidate.candidate_id}"
                for implementation, candidate in sorted(winners.items())
            )
        )
        attempt_root = paths.sweep_dir / "winner-quality" / signature
        tensor_quality_dir = attempt_root / "tensor-quality"
        product_output_dir = attempt_root / "product-output"
        inference_report = tensor_quality_dir / "inference-parity.json"
        preprocessing_report = tensor_quality_dir / "preprocessing-diagnostic.json"
        product_report = product_output_dir / "product-output-parity.json"
        if attempt_root.exists() and any(attempt_root.iterdir()):
            raise TuningWorkflowError(
                "Winner quality attempt already exists; remove only this partial "
                f"attempt before retrying: {attempt_root}"
            )
        base = _base_make_variables(
            paths,
            variant=args.variant,
            engine=engine,
            vsgan_engine=vsgan_engine,
            gpu_id=args.gpu_id,
        )
        variables = {
            **base,
            "VSTRT_ARGS": winners["vstrt"].runner_arguments(),
            "VSGAN_ARGS": winners["vsgan"].runner_arguments(),
            "MODEL_SPACE_DIR": paths.relative(tensor_quality_dir),
            "PRODUCT_OUTPUT_DIR": paths.relative(product_output_dir),
        }
        _progress(f"[tuned quality attempt {attempt}] Tensor-space quality")
        inference_result = runner.run(
            "tensor-quality",
            variables,
            accepted_artifact=inference_report,
        )
        if inference_result != 0:
            failures = _failed_inference_implementations(inference_report)
            if not failures:
                raise TuningWorkflowError(
                    "Winner inference gate failed without a candidate-specific "
                    f"failure: {inference_report}"
                )
            _record_disqualifications(
                disqualifications_path,
                winners=winners,
                failures=failures,
                evidence=inference_report,
                paths=paths,
            )
            continue

        _progress(f"[tuned quality attempt {attempt}] Product-output gate")
        product_result = runner.run(
            "product-output-parity",
            variables,
            accepted_artifact=product_report,
        )
        if product_result != 0:
            common_failure = _common_product_output_failure(product_report)
            if common_failure is not None:
                raise TuningWorkflowError(common_failure)
            failures = _failed_implementations(product_report)
            if not failures:
                raise TuningWorkflowError(
                    "Winner product-output gate failed without a candidate-specific "
                    f"failure: {product_report}"
                )
            _record_disqualifications(
                disqualifications_path,
                winners=winners,
                failures=failures,
                evidence=product_report,
                paths=paths,
            )
            continue

        final_report = {
            "schema_version": 1,
            "document_type": "tuned-winner-quality",
            "status": "valid",
            "publishable": False,
            "scope": "winner-quality",
            "workload_id": workload["id"],
            "variant": args.variant,
            "selection": {
                "path": paths.relative(paths.sweep_dir / "selection.json"),
                "sha256": _sha256(paths.sweep_dir / "selection.json"),
            },
            "winners": selection["winners"],
            "quality": {
                "inference_parity": {
                    "path": paths.relative(inference_report),
                    "sha256": _sha256(inference_report),
                },
                "preprocessing_diagnostic": {
                    "path": paths.relative(preprocessing_report),
                    "sha256": _sha256(preprocessing_report),
                },
                "product_output": {
                    "path": paths.relative(product_report),
                    "sha256": _sha256(product_report),
                },
            },
        }
        _write_json(paths.sweep_dir / "final-quality.json", final_report)
        _progress(f"[tuned quality attempt {attempt}] Winner quality valid")
        return final_report
    raise TuningWorkflowError("No tuned candidate pair passed the full quality gate")


def run_winner_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run the rotated campaign for the pair that passed full quality."""
    paths = _workflow_paths(args)
    contract = load_tuning_contract(
        _required_file(paths, Path(args.contract), label="Tuning contract")
    )
    quality_path = paths.sweep_dir / "final-quality.json"
    quality = _load_json(quality_path)
    if quality.get("status") != "valid" or quality.get("variant") != args.variant:
        raise TuningWorkflowError("Full tuned winner quality is not valid")
    selection = _load_json(paths.sweep_dir / "selection.json")
    winners = _selected_candidates(selection, contract)
    signature = _winner_signature(winners)
    quality_values = quality["quality"]
    inference_report = _under_root(
        paths,
        Path(quality_values["inference_parity"]["path"]),
        label="Inference parity report",
    )
    preprocessing_report = _under_root(
        paths,
        Path(quality_values["preprocessing_diagnostic"]["path"]),
        label="Preprocessing diagnostic",
    )
    product_report = _under_root(
        paths,
        Path(quality_values["product_output"]["path"]),
        label="Product-output report",
    )
    engine = _required_file(paths, Path(args.engine), label="Engine")
    vsgan_engine = _required_file(
        paths,
        Path(args.vsgan_engine),
        label="VSGAN engine",
    )
    campaign_dir = paths.sweep_dir / "winner-campaign" / signature
    variables = {
        **_base_make_variables(
            paths,
            variant=args.variant,
            engine=engine,
            vsgan_engine=vsgan_engine,
            gpu_id=args.gpu_id,
        ),
        "VSTRT_ARGS": winners["vstrt"].runner_arguments(),
        "VSGAN_ARGS": winners["vsgan"].runner_arguments(),
        "MODEL_SPACE_DIR": paths.relative(inference_report.parent),
        "PRODUCT_OUTPUT_DIR": paths.relative(product_report.parent),
        "CAMPAIGN_DIR": paths.relative(campaign_dir),
        "RESUME": "1" if args.resume else "0",
    }
    if preprocessing_report.parent != inference_report.parent:
        raise TuningWorkflowError("Tensor-space quality reports are not co-located")
    _progress(
        "[tuned campaign] "
        + ", ".join(
            f"{implementation}={candidate.candidate_id}"
            for implementation, candidate in sorted(winners.items())
        )
    )
    MakeRunner(paths, executable=args.make).run("run-comparative", variables)
    campaign_path = campaign_dir / "campaign.json"
    campaign = _load_json(campaign_path)
    if (
        campaign.get("status") != "valid"
        or campaign.get("publishable") is not True
        or campaign.get("execution_profile") != "tuned"
    ):
        raise TuningWorkflowError(f"Tuned winner campaign is not publishable: {campaign_path}")
    report = {
        "schema_version": 1,
        "document_type": "tuned-winner-campaign",
        "status": "valid",
        "publishable": False,
        "scope": "single-resolution-evidence",
        "workload_id": campaign["workload_id"],
        "variant": campaign["variant"],
        "winners": selection["winners"],
        "quality": quality["quality"],
        "campaign": {
            "path": paths.relative(campaign_path),
            "sha256": _sha256(campaign_path),
        },
    }
    _write_json(paths.sweep_dir / "final-campaign.json", report)
    return report


def _under_root(paths: WorkflowPaths, path: Path, *, label: str) -> Path:
    resolved = path.resolve() if path.is_absolute() else (paths.root / path).resolve()
    if resolved != paths.root and paths.root not in resolved.parents:
        raise TuningWorkflowError(f"{label} escapes the repository root")
    return resolved


def _required_file(paths: WorkflowPaths, path: Path, *, label: str) -> Path:
    resolved = _under_root(paths, path, label=label)
    if not resolved.is_file():
        raise TuningWorkflowError(f"{label} not found: {resolved}")
    return resolved


def _workflow_paths(args: argparse.Namespace) -> WorkflowPaths:
    return WorkflowPaths.resolve(
        root=Path(args.root),
        benchmarks_dir=Path(args.benchmarks_dir),
        manifest=Path(args.manifest),
        sweep_dir=Path(args.sweep_dir),
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", default="benchmarks/tuning/candidates.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--vsgan-engine", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks-dir", default="benchmarks")
    parser.add_argument("--make", default="make")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser("sweep")
    _add_common_arguments(sweep)
    sweep.add_argument("--resume", action="store_true")
    sweep.set_defaults(handler=run_sweep)

    quality = subparsers.add_parser("quality")
    _add_common_arguments(quality)
    quality.set_defaults(handler=run_winner_quality)

    campaign = subparsers.add_parser("campaign")
    _add_common_arguments(campaign)
    campaign.add_argument("--resume", action="store_true")
    campaign.set_defaults(handler=run_winner_campaign)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = args.handler(args)
    except (
        TuningContractError,
        TuningEvidenceError,
        TuningWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(
        f"{report['document_type']} {report['status']}: "
        f"{report['workload_id']} {report['variant']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
