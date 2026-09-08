"""Pure decision rules for the two-stage adaptive tuned search."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median
from typing import Any

from benchmarks.scripts.tuning.contract import Candidate, TasCandidate, TunedCandidate


@dataclass(frozen=True)
class CandidatePoint:
    """One measured scheduling point used by a search decision."""

    candidate: Candidate
    median_fps: float
    relative_spread: float
    suite_path: str
    median_peak_vram_mib: float | None = None
    median_cpu_cores: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "candidate_id": self.candidate.candidate_id,
            "cuda_graph": self.candidate.cuda_graph,
            "median_fps": self.median_fps,
            "relative_spread": self.relative_spread,
            "suite": self.suite_path,
        }
        if isinstance(self.candidate, TunedCandidate):
            result["num_streams"] = self.candidate.num_streams
        else:
            result.update(
                decode_method=self.candidate.decode_method,
                writer=self.candidate.writer,
                median_peak_vram_mib=self.median_peak_vram_mib,
                median_cpu_cores=self.median_cpu_cores,
            )
        return result


def stream_count(candidate: Candidate) -> int:
    """Guard the stream-only decision rules against finite-grid candidates."""
    if not isinstance(candidate, TunedCandidate):
        raise ValueError("Stream decision rules only apply to vstrt")
    return candidate.num_streams


def resource_medians(manifests: Iterable[dict[str, Any]]) -> tuple[float, float]:
    """Derive resource tie-break inputs from raw runs, not suite summary claims."""
    peaks = []
    cores = []
    for manifest in manifests:
        try:
            metrics = manifest["measured"]["metrics"]
            peak = metrics["nvml"]["memory"]["peak_delta_mib"]
            cpu = metrics["cpu"]["average_cores"]
        except (KeyError, TypeError) as exc:
            raise ValueError("TAS run is missing peak VRAM or CPU metrics") from exc
        for value in (peak, cpu):
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("TAS resource metrics must be finite non-negative numbers")
        peaks.append(float(peak))
        cores.append(float(cpu))
    if not peaks:
        raise ValueError("TAS resource tie-break requires run manifests")
    return float(median(peaks)), float(median(cores))


def resource_order(point: CandidatePoint) -> tuple[float, float, str]:
    if isinstance(point.candidate, TasCandidate):
        if point.median_peak_vram_mib is None or point.median_cpu_cores is None:
            raise ValueError("TAS resource tie-break requires peak VRAM and CPU medians")
        for value in (point.median_peak_vram_mib, point.median_cpu_cores):
            if not math.isfinite(value) or value < 0:
                raise ValueError("TAS resource medians must be finite and non-negative")
        return point.median_peak_vram_mib, point.median_cpu_cores, point.candidate.candidate_id
    return (
        point.candidate.num_streams,
        int(point.candidate.cuda_graph),
        point.candidate.candidate_id,
    )


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
    ordered = sorted(points, key=lambda point: stream_count(point.candidate))
    previous_best = max(point.median_fps for point in ordered[:-1])
    return ordered[-1].median_fps > previous_best * (1 + relative_margin)


def shortlist(
    points: Iterable[CandidatePoint],
    *,
    size: int,
) -> tuple[Candidate, ...]:
    """Select the strongest reconnaissance points without reusing their statistics."""
    ordered = sorted(
        points,
        key=lambda point: (
            -point.median_fps,
            stream_count(point.candidate) if isinstance(point.candidate, TunedCandidate) else 0,
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
        key=resource_order,
    )
