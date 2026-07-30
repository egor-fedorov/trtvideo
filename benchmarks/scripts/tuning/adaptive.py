"""Pure decision rules for the two-stage adaptive tuned search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmarks.scripts.tuning.contract import TunedCandidate


@dataclass(frozen=True)
class CandidatePoint:
    """One measured scheduling point used by a search decision."""

    candidate: TunedCandidate
    median_fps: float
    relative_spread: float
    suite_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "num_streams": self.candidate.num_streams,
            "cuda_graph": self.candidate.cuda_graph,
            "median_fps": self.median_fps,
            "relative_spread": self.relative_spread,
            "suite": self.suite_path,
        }


def has_confirmed_decline(
    points: list[CandidatePoint],
    *,
    relative_margin: float,
    patience: int,
) -> bool:
    """Return whether the trailing points are materially below the best seen."""
    if len(points) <= patience:
        return False
    best = max(point.median_fps for point in points)
    threshold = best * (1 - relative_margin)
    return all(point.median_fps < threshold for point in points[-patience:])


def sentinel_recovers(
    points: list[CandidatePoint],
    sentinel: CandidatePoint,
    *,
    relative_margin: float,
) -> bool:
    """Return whether the maximum-range sentinel invalidates an early stop."""
    best = max(point.median_fps for point in points)
    return sentinel.median_fps > best * (1 + relative_margin)


def upper_boundary_unresolved(
    points: list[CandidatePoint],
    *,
    relative_margin: float,
) -> bool:
    """Return whether the maximum stream count is still materially improving."""
    if len(points) < 2:
        return True
    ordered = sorted(points, key=lambda point: point.candidate.num_streams)
    previous_best = max(point.median_fps for point in ordered[:-1])
    return ordered[-1].median_fps > previous_best * (1 + relative_margin)


def shortlist(
    points: Iterable[CandidatePoint],
    *,
    size: int,
) -> tuple[TunedCandidate, ...]:
    """Select the strongest reconnaissance points without reusing their statistics."""
    ordered = sorted(
        points,
        key=lambda point: (
            -point.median_fps,
            point.candidate.num_streams,
            point.candidate.candidate_id,
        ),
    )
    return tuple(point.candidate for point in ordered[:size])


def select_peak_equivalent(
    points: Iterable[CandidatePoint],
    *,
    equivalence_margin: float,
) -> CandidatePoint | None:
    """Select the cheapest point that remains within the declared peak margin."""
    eligible = list(points)
    if not eligible:
        return None
    maximum = max(point.median_fps for point in eligible)
    threshold = maximum * (1 - equivalence_margin)
    equivalent = [point for point in eligible if point.median_fps >= threshold]
    return min(
        equivalent,
        key=lambda point: (
            point.candidate.num_streams,
            point.candidate.cuda_graph,
            point.candidate.candidate_id,
        ),
    )
