"""Canonical adaptive tuning contract and CLI argument rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Implementation = Literal["vstrt", "vsgan"]
AutoOrInt = Literal["auto"] | int

_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_IMPLEMENTATIONS: tuple[Implementation, ...] = ("vstrt", "vsgan")


class TuningContractError(ValueError):
    """Raised when the tuned-search contract is invalid."""


def _auto_or_positive_int(value: Any, *, field: str) -> AutoOrInt:
    if value == "auto":
        return "auto"
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TuningContractError(f"{field} must be 'auto' or a positive integer")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TuningContractError(f"{field} must be a positive integer")
    return value


def _fraction(value: Any, *, field: str, allow_zero: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        qualifier = "[0, 1)" if allow_zero else "(0, 1)"
        raise TuningContractError(f"{field} must be in {qualifier}")
    lower_ok = value >= 0 if allow_zero else value > 0
    if not lower_ok or value >= 1:
        qualifier = "[0, 1)" if allow_zero else "(0, 1)"
        raise TuningContractError(f"{field} must be in {qualifier}")
    return float(value)


@dataclass(frozen=True)
class TunedCandidate:
    """One explicit VapourSynth scheduling point."""

    candidate_id: str
    implementation: Implementation
    requests: AutoOrInt
    num_streams: int
    vapoursynth_threads: AutoOrInt
    cuda_graph: bool

    def __post_init__(self) -> None:
        if not _CANDIDATE_ID_RE.fullmatch(self.candidate_id):
            raise TuningContractError(
                "Candidate id must contain only lowercase letters, digits, and hyphens"
            )

    def execution_profile(self) -> dict[str, str | int | bool]:
        """Return the exact profile stored by benchmark runners."""
        return {
            "execution_profile": "tuned",
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
class MeasurementPolicy:
    """One performance-measurement stage of the tuned search."""

    measured_frames: int
    warmup_frames: int
    initial_runs: int
    extra_runs_on_spread: int
    spread_threshold: float
    max_relative_spread: float
    idle_seconds: float
    bitrate_validation: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, field: str) -> MeasurementPolicy:
        extra_runs = value.get("extra_runs_on_spread")
        idle_seconds = value.get("idle_seconds")
        bitrate_validation = value.get("bitrate_validation")
        if not isinstance(extra_runs, int) or isinstance(extra_runs, bool) or extra_runs < 0:
            raise TuningContractError(f"{field}.extra_runs_on_spread must be non-negative")
        if (
            not isinstance(idle_seconds, (int, float))
            or isinstance(idle_seconds, bool)
            or idle_seconds < 0
        ):
            raise TuningContractError(f"{field}.idle_seconds must be non-negative")
        if not isinstance(bitrate_validation, bool):
            raise TuningContractError(f"{field}.bitrate_validation must be boolean")
        spread_threshold = _fraction(
            value.get("spread_threshold"),
            field=f"{field}.spread_threshold",
        )
        max_relative_spread = _fraction(
            value.get("max_relative_spread"),
            field=f"{field}.max_relative_spread",
        )
        if spread_threshold > max_relative_spread:
            raise TuningContractError(f"{field}.spread_threshold cannot exceed max_relative_spread")
        return cls(
            measured_frames=_positive_int(
                value.get("measured_frames"),
                field=f"{field}.measured_frames",
            ),
            warmup_frames=_positive_int(
                value.get("warmup_frames"),
                field=f"{field}.warmup_frames",
            ),
            initial_runs=_positive_int(
                value.get("initial_runs"),
                field=f"{field}.initial_runs",
            ),
            extra_runs_on_spread=extra_runs,
            spread_threshold=spread_threshold,
            max_relative_spread=max_relative_spread,
            idle_seconds=float(idle_seconds),
            bitrate_validation=bitrate_validation,
        )

    def runner_arguments(self) -> str:
        """Render runner overrides for this search stage."""
        arguments = [
            f"--frames {self.measured_frames}",
            f"--warmup-frames {self.warmup_frames}",
            f"--runs {self.initial_runs}",
            f"--extra-runs {self.extra_runs_on_spread}",
            f"--spread-threshold {self.spread_threshold}",
            f"--max-relative-spread {self.max_relative_spread}",
            f"--idle-seconds {self.idle_seconds:g}",
        ]
        if not self.bitrate_validation:
            arguments.append("--skip-bitrate-validation")
        return " ".join(arguments)

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "measured_frames": self.measured_frames,
            "warmup_frames": self.warmup_frames,
            "initial_runs": self.initial_runs,
            "extra_runs_on_spread": self.extra_runs_on_spread,
            "spread_threshold": self.spread_threshold,
            "max_relative_spread": self.max_relative_spread,
            "idle_seconds": self.idle_seconds,
            "bitrate_validation": self.bitrate_validation,
        }


@dataclass(frozen=True)
class SearchPolicy:
    """Predeclared adaptive reconnaissance and confirmation rules."""

    minimum_streams: int
    maximum_streams: int
    decline_margin: float
    decline_patience: int
    sentinel_streams: int
    shortlist_size: int
    reconnaissance: MeasurementPolicy
    confirmation: MeasurementPolicy

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SearchPolicy:
        minimum = _positive_int(value.get("minimum_streams"), field="search.minimum_streams")
        maximum = _positive_int(value.get("maximum_streams"), field="search.maximum_streams")
        sentinel = _positive_int(value.get("sentinel_streams"), field="search.sentinel_streams")
        if minimum >= maximum:
            raise TuningContractError("search.minimum_streams must be below maximum_streams")
        if sentinel != maximum:
            raise TuningContractError("search.sentinel_streams must equal maximum_streams")
        reconnaissance = value.get("reconnaissance")
        confirmation = value.get("confirmation")
        if not isinstance(reconnaissance, dict) or not isinstance(confirmation, dict):
            raise TuningContractError(
                "search.reconnaissance and search.confirmation must be objects"
            )
        return cls(
            minimum_streams=minimum,
            maximum_streams=maximum,
            decline_margin=_fraction(
                value.get("decline_margin"),
                field="search.decline_margin",
                allow_zero=False,
            ),
            decline_patience=_positive_int(
                value.get("decline_patience"),
                field="search.decline_patience",
            ),
            sentinel_streams=sentinel,
            shortlist_size=_positive_int(
                value.get("shortlist_size"),
                field="search.shortlist_size",
            ),
            reconnaissance=MeasurementPolicy.from_dict(
                reconnaissance,
                field="search.reconnaissance",
            ),
            confirmation=MeasurementPolicy.from_dict(
                confirmation,
                field="search.confirmation",
            ),
        )

    @property
    def stream_range(self) -> range:
        return range(self.minimum_streams, self.maximum_streams + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_streams": self.minimum_streams,
            "maximum_streams": self.maximum_streams,
            "decline_margin": self.decline_margin,
            "decline_patience": self.decline_patience,
            "sentinel_streams": self.sentinel_streams,
            "shortlist_size": self.shortlist_size,
            "reconnaissance": self.reconnaissance.as_dict(),
            "confirmation": self.confirmation.as_dict(),
        }


@dataclass(frozen=True)
class SelectionPolicy:
    """Predeclared winner-selection rule."""

    metric: str
    equivalence_margin: float
    max_relative_spread: float
    tie_breaker: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SelectionPolicy:
        if value.get("metric") != "median_end_to_end_fps":
            raise TuningContractError("Selection metric must be 'median_end_to_end_fps'")
        if value.get("tie_breaker") != "lowest_num_streams_then_graph_off":
            raise TuningContractError(
                "Selection tie_breaker must be 'lowest_num_streams_then_graph_off'"
            )
        return cls(
            metric="median_end_to_end_fps",
            equivalence_margin=_fraction(
                value.get("equivalence_margin"),
                field="selection.equivalence_margin",
                allow_zero=False,
            ),
            max_relative_spread=_fraction(
                value.get("max_relative_spread"),
                field="selection.max_relative_spread",
            ),
            tie_breaker="lowest_num_streams_then_graph_off",
        )

    def as_dict(self) -> dict[str, str | float]:
        return {
            "metric": self.metric,
            "equivalence_margin": self.equivalence_margin,
            "max_relative_spread": self.max_relative_spread,
            "tie_breaker": self.tie_breaker,
        }


@dataclass(frozen=True)
class ImplementationProfile:
    """Scheduling values fixed while one competitor is searched."""

    implementation: Implementation
    requests: AutoOrInt
    vapoursynth_threads: AutoOrInt
    probe_cuda_graph: bool

    @classmethod
    def from_dict(
        cls,
        implementation: Implementation,
        value: dict[str, Any],
    ) -> ImplementationProfile:
        probe_cuda_graph = value.get("probe_cuda_graph")
        if not isinstance(probe_cuda_graph, bool):
            raise TuningContractError(
                f"implementations.{implementation}.probe_cuda_graph must be boolean"
            )
        return cls(
            implementation=implementation,
            requests=_auto_or_positive_int(
                value.get("requests"),
                field=f"implementations.{implementation}.requests",
            ),
            vapoursynth_threads=_auto_or_positive_int(
                value.get("vapoursynth_threads"),
                field=f"implementations.{implementation}.vapoursynth_threads",
            ),
            probe_cuda_graph=probe_cuda_graph,
        )


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
    """Validated adaptive search and winner-selection contract."""

    schema_version: int
    selection: SelectionPolicy
    search: SearchPolicy
    project_profile: ProjectProfile
    implementations: tuple[ImplementationProfile, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuningContract:
        if value.get("schema_version") != 2:
            raise TuningContractError("Unsupported tuning contract schema_version")
        selection_value = value.get("selection")
        search_value = value.get("search")
        project_profile_value = value.get("project_profile")
        implementations_value = value.get("implementations")
        if not isinstance(selection_value, dict):
            raise TuningContractError("Tuning selection must be an object")
        if not isinstance(search_value, dict):
            raise TuningContractError("Tuning search must be an object")
        if not isinstance(project_profile_value, dict):
            raise TuningContractError("Tuning project_profile must be an object")
        if not isinstance(implementations_value, dict):
            raise TuningContractError("Tuning implementations must be an object")
        if set(implementations_value) != set(_IMPLEMENTATIONS):
            raise TuningContractError("Tuning contract must define vstrt and vsgan")
        implementations = tuple(
            ImplementationProfile.from_dict(implementation, implementations_value[implementation])
            for implementation in _IMPLEMENTATIONS
            if isinstance(implementations_value[implementation], dict)
        )
        if len(implementations) != len(_IMPLEMENTATIONS):
            raise TuningContractError("Every tuning implementation must be an object")
        selection = SelectionPolicy.from_dict(selection_value)
        search = SearchPolicy.from_dict(search_value)
        if selection.max_relative_spread != search.confirmation.max_relative_spread:
            raise TuningContractError("Selection and confirmation max_relative_spread must match")
        return cls(
            schema_version=2,
            selection=selection,
            search=search,
            project_profile=ProjectProfile.from_dict(project_profile_value),
            implementations=implementations,
        )

    @property
    def candidates(self) -> tuple[TunedCandidate, ...]:
        candidates = []
        for profile in self.implementations:
            for streams in self.search.stream_range:
                candidates.append(self.make_candidate(profile.implementation, streams))
                if profile.probe_cuda_graph:
                    candidates.append(
                        self.make_candidate(
                            profile.implementation,
                            streams,
                            cuda_graph=True,
                        )
                    )
        return tuple(candidates)

    def implementation(self, name: Implementation) -> ImplementationProfile:
        return next(profile for profile in self.implementations if profile.implementation == name)

    def make_candidate(
        self,
        implementation: Implementation,
        num_streams: int,
        *,
        cuda_graph: bool = False,
    ) -> TunedCandidate:
        if num_streams not in self.search.stream_range:
            raise TuningContractError(
                f"Candidate streams must be in "
                f"{self.search.minimum_streams}..{self.search.maximum_streams}"
            )
        profile = self.implementation(implementation)
        if cuda_graph and not profile.probe_cuda_graph:
            raise TuningContractError(f"{implementation} does not allow a CUDA Graph probe")
        thread_suffix = "-tauto" if implementation == "vsgan" else ""
        candidate_id = f"{implementation}-s{num_streams}{thread_suffix}-g{int(cuda_graph)}"
        return TunedCandidate(
            candidate_id=candidate_id,
            implementation=implementation,
            requests=profile.requests,
            num_streams=num_streams,
            vapoursynth_threads=profile.vapoursynth_threads,
            cuda_graph=cuda_graph,
        )

    def for_implementation(
        self,
        implementation: Implementation,
    ) -> tuple[TunedCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate.implementation == implementation
        )

    def candidate(self, candidate_id: str) -> TunedCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise TuningContractError(f"Unknown tuned candidate: {candidate_id}")


def load_tuning_contract(path: Path) -> TuningContract:
    """Load and validate an adaptive tuning JSON contract."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningContractError(f"Cannot read tuning contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TuningContractError("Tuning contract root must be an object")
    return TuningContract.from_dict(value)
