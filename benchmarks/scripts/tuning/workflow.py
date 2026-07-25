"""Run the declared tuned sweep and finalize its selected candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.scripts.tuning.contract import (
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
from benchmarks.scripts.workloads.manifest import load_manifest


class TuningWorkflowError(RuntimeError):
    """Raised when a tuned workflow cannot preserve its evidence contract."""


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
            resolved = (
                path.resolve()
                if path.is_absolute()
                else (resolved_root / path).resolve()
            )
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
            raise TuningWorkflowError(
                f"Artifact is outside the repository root: {path}"
            ) from exc


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
        raise TuningWorkflowError(
            f"Partial evidence must be removed before retrying: {directory}"
        )
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
        "VAPOURSYNTH_MODE": "tuned",
    }


def _candidate_variables(
    candidate: TunedCandidate,
    base: dict[str, str],
) -> dict[str, str]:
    values = dict(base)
    values[
        "VSTRT_ARGS" if candidate.implementation == "vstrt" else "VSGAN_ARGS"
    ] = candidate.runner_arguments()
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


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Measure and validate every declared tuned candidate."""
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
    if (
        paths.sweep_dir.exists()
        and any(paths.sweep_dir.iterdir())
        and not args.resume
    ):
        raise TuningWorkflowError(
            f"Sweep directory is not empty; use --resume: {paths.sweep_dir}"
        )
    paths.sweep_dir.mkdir(parents=True, exist_ok=True)
    base = _base_make_variables(
        paths,
        variant=args.variant,
        engine=engine,
        vsgan_engine=vsgan_engine,
        gpu_id=args.gpu_id,
    )

    reference_root = paths.sweep_dir / "reference" / "model-space"
    reference_manifest = reference_root / "ai-media" / "manifest.json"
    if _require_clean_destination(
        reference_root,
        marker=reference_manifest,
        resume=args.resume,
    ):
        runner.run(
            "capture-model-ai-media",
            {
                **base,
                "MODEL_SPACE_DIR": paths.relative(reference_root),
            },
        )

    for candidate in contract.candidates:
        candidate_root = candidate_directory(paths.sweep_dir, candidate)
        variables = _candidate_variables(candidate, base)
        performance_dir = candidate_root / "performance"
        suite_path = performance_dir / "suite.json"
        if _require_clean_destination(
            performance_dir,
            marker=suite_path,
            resume=args.resume,
        ):
            output_variable = (
                "VSTRT_OUTPUT_DIR"
                if candidate.implementation == "vstrt"
                else "VSGAN_OUTPUT_DIR"
            )
            runner.run(
                f"run-{candidate.implementation}",
                {
                    **variables,
                    output_variable: paths.relative(performance_dir),
                    "ARGS": "",
                },
                accepted_artifact=suite_path,
            )

        capture_root = candidate_root / "model-space"
        capture_manifest = (
            capture_root / candidate.implementation / "manifest.json"
        )
        if _require_clean_destination(
            capture_root,
            marker=capture_manifest,
            resume=args.resume,
        ):
            runner.run(
                f"capture-model-{candidate.implementation}",
                {
                    **variables,
                    "MODEL_SPACE_DIR": paths.relative(capture_root),
                },
            )

        report_path = candidate_root / "model-space-parity.json"
        if report_path.is_file() and not args.resume:
            raise TuningWorkflowError(
                f"Evidence already exists; use --resume: {report_path}"
            )
        if not report_path.is_file():
            runner.run(
                "compare-model-space-candidate",
                {
                    **base,
                    "MODEL_SPACE_REFERENCE": paths.relative(reference_manifest),
                    "MODEL_SPACE_CANDIDATE": paths.relative(capture_manifest),
                    "MODEL_SPACE_REPORT": paths.relative(report_path),
                },
                accepted_artifact=report_path,
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
            raise TuningWorkflowError(
                f"Tuned candidate selection has no {implementation} winner"
            )
        result[implementation] = contract.candidate(candidate_id)
    return result


def _winner_signature(winners: dict[str, TunedCandidate]) -> str:
    return "__".join(
        winners[implementation].candidate_id for implementation in sorted(winners)
    )


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


def _record_disqualifications(
    path: Path,
    *,
    winners: dict[str, TunedCandidate],
    failures: dict[str, str],
    evidence: Path,
    paths: WorkflowPaths,
) -> None:
    value = (
        _load_json(path)
        if path.is_file()
        else {"schema_version": 1, "entries": []}
    )
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

    for _ in range(len(contract.candidates)):
        selection = _write_selection(
            contract=contract,
            workload=workload,
            variant=args.variant,
            paths=paths,
            disqualifications_path=disqualifications_path,
        )
        winners = _selected_candidates(selection, contract)
        signature = _winner_signature(winners)
        attempt_root = paths.sweep_dir / "winner-quality" / signature
        model_space_dir = attempt_root / "model-space"
        product_output_dir = attempt_root / "product-output"
        model_report = model_space_dir / "model-space-parity.json"
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
            "MODEL_SPACE_DIR": paths.relative(model_space_dir),
            "PRODUCT_OUTPUT_DIR": paths.relative(product_output_dir),
        }
        model_result = runner.run(
            "model-space-parity",
            variables,
            accepted_artifact=model_report,
        )
        if model_result != 0:
            failures = _failed_implementations(model_report)
            if not failures:
                raise TuningWorkflowError(
                    "Winner model-space gate failed without a candidate-specific "
                    f"failure: {model_report}"
                )
            _record_disqualifications(
                disqualifications_path,
                winners=winners,
                failures=failures,
                evidence=model_report,
                paths=paths,
            )
            continue

        product_result = runner.run(
            "product-output-parity",
            variables,
            accepted_artifact=product_report,
        )
        if product_result != 0:
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
                "model_space": {
                    "path": paths.relative(model_report),
                    "sha256": _sha256(model_report),
                },
                "product_output": {
                    "path": paths.relative(product_report),
                    "sha256": _sha256(product_report),
                },
            },
        }
        _write_json(paths.sweep_dir / "final-quality.json", final_report)
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
    model_report = _under_root(
        paths,
        Path(quality_values["model_space"]["path"]),
        label="Model-space report",
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
        "MODEL_SPACE_DIR": paths.relative(model_report.parent),
        "PRODUCT_OUTPUT_DIR": paths.relative(product_report.parent),
        "CAMPAIGN_DIR": paths.relative(campaign_dir),
        "RESUME": "1" if args.resume else "0",
    }
    MakeRunner(paths, executable=args.make).run("run-comparative", variables)
    campaign_path = campaign_dir / "campaign.json"
    campaign = _load_json(campaign_path)
    if (
        campaign.get("status") != "valid"
        or campaign.get("publishable") is not True
        or campaign.get("comparison_profile") != "tuned"
    ):
        raise TuningWorkflowError(
            f"Tuned winner campaign is not publishable: {campaign_path}"
        )
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
        f"{report['workload_id']} {report['variant']}"
    )


if __name__ == "__main__":
    main()
