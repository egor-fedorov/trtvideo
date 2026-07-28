"""Canonical tuned-candidate contract and CLI argument rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Implementation = Literal["vstrt", "vsgan"]
AutoOrInt = Literal["auto"] | int

_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_IMPLEMENTATIONS = ("vstrt", "vsgan")
_REQUIRED_AUTO_STREAM_SWEEPS = {
    "vstrt": {2, 3, 4},
    "vsgan": {2, 3, 4, 5, 6},
}


class TuningContractError(ValueError):
    """Raised when the tuned-candidate contract is invalid."""


def _auto_or_positive_int(value: Any, *, field: str) -> AutoOrInt:
    if value == "auto":
        return "auto"
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TuningContractError(f"{field} must be 'auto' or a positive integer")
    return value


@dataclass(frozen=True)
class TunedCandidate:
    """One explicit VapourSynth scheduling point."""

    candidate_id: str
    implementation: Implementation
    requests: AutoOrInt
    num_streams: int
    vapoursynth_threads: AutoOrInt
    cuda_graph: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TunedCandidate:
        candidate_id = value.get("id")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(
            candidate_id
        ):
            raise TuningContractError(
                "Candidate id must contain only lowercase letters, digits, and hyphens"
            )
        implementation = value.get("implementation")
        if implementation not in _IMPLEMENTATIONS:
            raise TuningContractError(
                f"Candidate {candidate_id} has an invalid implementation"
            )
        num_streams = value.get("num_streams")
        if (
            not isinstance(num_streams, int)
            or isinstance(num_streams, bool)
            or num_streams <= 0
        ):
            raise TuningContractError(
                f"Candidate {candidate_id} num_streams must be positive"
            )
        cuda_graph = value.get("cuda_graph")
        if not isinstance(cuda_graph, bool):
            raise TuningContractError(
                f"Candidate {candidate_id} cuda_graph must be boolean"
            )
        return cls(
            candidate_id=candidate_id,
            implementation=implementation,
            requests=_auto_or_positive_int(
                value.get("requests"),
                field=f"Candidate {candidate_id} requests",
            ),
            num_streams=num_streams,
            vapoursynth_threads=_auto_or_positive_int(
                value.get("vapoursynth_threads"),
                field=f"Candidate {candidate_id} vapoursynth_threads",
            ),
            cuda_graph=cuda_graph,
        )

    def execution_profile(self) -> dict[str, str | int | bool]:
        """Return the exact profile stored by benchmark runners."""
        return {
            "mode": "tuned",
            "vspipe_requests": self.requests,
            "num_streams": self.num_streams,
            "vapoursynth_threads": self.vapoursynth_threads,
            "cuda_graph": self.cuda_graph,
        }

    def runner_arguments(self) -> str:
        """Render an immutable runner argument string for Make."""
        graph_option = "--cuda-graph" if self.cuda_graph else "--no-cuda-graph"
        return " ".join(
            (
                f"--requests {self.requests}",
                f"--num-streams {self.num_streams}",
                f"--vs-threads {self.vapoursynth_threads}",
                graph_option,
            )
        )


@dataclass(frozen=True)
class SelectionPolicy:
    """Predeclared winner-selection rule."""

    metric: str
    max_relative_spread: float
    require_complete_sweep: bool
    tie_breaker: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SelectionPolicy:
        metric = value.get("metric")
        tie_breaker = value.get("tie_breaker")
        max_relative_spread = value.get("max_relative_spread")
        require_complete_sweep = value.get("require_complete_sweep")
        if metric != "median_end_to_end_fps":
            raise TuningContractError(
                "Selection metric must be 'median_end_to_end_fps'"
            )
        if tie_breaker != "candidate_id":
            raise TuningContractError("Selection tie_breaker must be 'candidate_id'")
        if (
            not isinstance(max_relative_spread, (int, float))
            or isinstance(max_relative_spread, bool)
            or not 0 <= max_relative_spread < 1
        ):
            raise TuningContractError(
                "Selection max_relative_spread must be in [0, 1)"
            )
        if not isinstance(require_complete_sweep, bool):
            raise TuningContractError(
                "Selection require_complete_sweep must be boolean"
            )
        return cls(
            metric=metric,
            max_relative_spread=float(max_relative_spread),
            require_complete_sweep=require_complete_sweep,
            tie_breaker=tie_breaker,
        )

    def as_dict(self) -> dict[str, str | float | bool]:
        return {
            "metric": self.metric,
            "max_relative_spread": self.max_relative_spread,
            "require_complete_sweep": self.require_complete_sweep,
            "tie_breaker": self.tie_breaker,
        }


@dataclass(frozen=True)
class ProjectProfile:
    """Fixed project configuration used while competitors are tuned."""

    backend: str
    cuda_graph: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectProfile:
        if value.get("backend") != "nvcodec":
            raise TuningContractError("Tuned project backend must be 'nvcodec'")
        if value.get("cuda_graph") is not False:
            raise TuningContractError(
                "Tuned project profile requires the verified CUDA Graph off state"
            )
        return cls(backend="nvcodec", cuda_graph=False)

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "backend": self.backend,
            "cuda_graph": self.cuda_graph,
        }


@dataclass(frozen=True)
class TuningContract:
    """Validated collection of all candidates that must be evaluated."""

    schema_version: int
    selection: SelectionPolicy
    project_profile: ProjectProfile
    candidates: tuple[TunedCandidate, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuningContract:
        if value.get("schema_version") != 1:
            raise TuningContractError("Unsupported tuning contract schema_version")
        selection_value = value.get("selection")
        project_profile_value = value.get("project_profile")
        candidates_value = value.get("candidates")
        if not isinstance(selection_value, dict):
            raise TuningContractError("Tuning selection must be an object")
        if not isinstance(project_profile_value, dict):
            raise TuningContractError("Tuning project_profile must be an object")
        if (
            not isinstance(candidates_value, list)
            or not candidates_value
            or not all(isinstance(item, dict) for item in candidates_value)
        ):
            raise TuningContractError(
                "Tuning candidates must be a non-empty object array"
            )
        candidates = tuple(
            TunedCandidate.from_dict(candidate) for candidate in candidates_value
        )
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise TuningContractError("Tuning candidate ids must be unique")
        present = {candidate.implementation for candidate in candidates}
        if present != set(_IMPLEMENTATIONS):
            raise TuningContractError(
                "Tuning contract must contain vstrt and vsgan candidates"
            )
        for implementation, required_streams in _REQUIRED_AUTO_STREAM_SWEEPS.items():
            actual_streams = {
                candidate.num_streams
                for candidate in candidates
                if candidate.implementation == implementation
                and candidate.requests == "auto"
                and candidate.vapoursynth_threads == "auto"
                and not candidate.cuda_graph
            }
            missing_streams = sorted(required_streams - actual_streams)
            if missing_streams:
                required = ", ".join(str(streams) for streams in sorted(required_streams))
                missing = ", ".join(str(streams) for streams in missing_streams)
                raise TuningContractError(
                    f"Tuning contract must sweep {implementation} streams {required} "
                    "with requests=auto, VapourSynth threads=auto, and CUDA Graph "
                    f"disabled; missing: {missing}"
                )
        return cls(
            schema_version=1,
            selection=SelectionPolicy.from_dict(selection_value),
            project_profile=ProjectProfile.from_dict(project_profile_value),
            candidates=candidates,
        )

    def for_implementation(
        self,
        implementation: Implementation,
    ) -> tuple[TunedCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.implementation == implementation
        )

    def candidate(self, candidate_id: str) -> TunedCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise TuningContractError(f"Unknown tuned candidate: {candidate_id}")


def load_tuning_contract(path: Path) -> TuningContract:
    """Load and validate a tuned-candidate JSON contract."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningContractError(f"Cannot read tuning contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TuningContractError("Tuning contract root must be an object")
    return TuningContract.from_dict(value)
