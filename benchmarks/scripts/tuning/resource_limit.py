"""Machine-verifiable resource-limit evidence for adaptive tuning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.manifest import (
    ManifestContractError,
    artifact_path,
    execution_profile,
    load_json,
)
from benchmarks.scripts.tuning.contract import MeasurementPolicy, TunedCandidate

_CUDA_OOM_MARKERS = (
    "Error Code 2: OutOfMemory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation",
)


class ResourceLimitError(RuntimeError):
    """Raised when resource-limit evidence differs from its search contract."""


@dataclass(frozen=True)
class ResourceLimitEvidence:
    """One failed run that proves a candidate exceeded available GPU memory."""

    kind: str
    suite_path: str
    run_manifest_path: str
    stderr_path: str
    stderr_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "suite": self.suite_path,
            "run_manifest": self.run_manifest_path,
            "stderr": self.stderr_path,
            "stderr_sha256": self.stderr_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def detect_cuda_oom(
    *,
    root: Path,
    suite_path: Path,
) -> ResourceLimitEvidence | None:
    """Return hashed CUDA OOM evidence from an invalid benchmark suite."""
    resolved_root = root.resolve()
    resolved_suite = suite_path.resolve()
    suite = load_json(resolved_suite)
    if suite.get("status") != "invalid":
        return None
    runs = suite.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        manifest_path = artifact_path(
            resolved_root,
            run.get("manifest"),
            label="resource-limit run manifest",
        )
        manifest = load_json(manifest_path)
        if manifest.get("status") != "invalid":
            continue
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for artifact_name in ("warmup_stderr", "measured_stderr"):
            value = artifacts.get(artifact_name)
            if not isinstance(value, str):
                continue
            stderr_path = artifact_path(
                resolved_root,
                value,
                label=f"resource-limit {artifact_name}",
            )
            if stderr_path.parent != manifest_path.parent or not stderr_path.is_file():
                continue
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            if not any(marker in stderr for marker in _CUDA_OOM_MARKERS):
                continue
            return ResourceLimitEvidence(
                kind="cuda-out-of-memory",
                suite_path=_relative(resolved_suite, resolved_root),
                run_manifest_path=_relative(manifest_path, resolved_root),
                stderr_path=_relative(stderr_path, resolved_root),
                stderr_sha256=_sha256(stderr_path),
            )
    return None


def validate_cuda_oom_record(
    record: dict[str, Any],
    *,
    candidate: TunedCandidate,
    policy: MeasurementPolicy,
    workload_id: str,
    variant: str,
    contract_version: int,
    root: Path,
    suite_path: Path,
) -> None:
    """Revalidate a persisted resource-ceiling record from raw artifacts."""
    try:
        suite = load_json(suite_path)
        parameters = suite.get("parameters")
        if not isinstance(parameters, dict):
            raise ResourceLimitError("Resource-limit suite has no parameters")
        expected_suite = {
            "status": "invalid",
            "workload_id": workload_id,
            "benchmark_contract_version": contract_version,
            "variant": variant,
        }
        changed = [key for key, value in expected_suite.items() if suite.get(key) != value]
        expected_parameters = {
            **candidate.execution_profile(),
            "frames": policy.measured_frames,
            "warmup_frames": policy.warmup_frames,
            "initial_runs": policy.initial_runs,
            "extra_runs_on_spread": policy.extra_runs_on_spread,
            "spread_threshold": policy.spread_threshold,
            "max_relative_spread": policy.max_relative_spread,
            "idle_seconds": policy.idle_seconds,
            "bitrate_validation": policy.bitrate_validation,
        }
        changed.extend(
            key for key, value in expected_parameters.items() if parameters.get(key) != value
        )
        if execution_profile(parameters) != candidate.execution_profile():
            changed.append("execution_profile")
        if changed:
            raise ResourceLimitError(
                "Resource-limit contract changed: " + ", ".join(sorted(set(changed)))
            )
        detected = detect_cuda_oom(root=root, suite_path=suite_path)
    except (ManifestContractError, OSError) as exc:
        raise ResourceLimitError(f"Invalid resource-limit artifact: {exc}") from exc
    expected_record = (
        {
            "candidate_id": candidate.candidate_id,
            "num_streams": candidate.num_streams,
            **detected.as_dict(),
        }
        if detected is not None
        else None
    )
    if record != expected_record:
        raise ResourceLimitError("Resource-limit evidence changed")
