"""Lifecycle timing boundaries shared by video benchmark runners."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, 2}
_CLOCK = "time.perf_counter_ns"
_BOUNDARY_CONTRACT = (
    "process_start -> first_frame_completed -> last_frame_completed -> process_exit"
)
_RESERVED_CHECKPOINTS = {
    "process_started",
    "first_frame_completed",
    "last_frame_completed",
    "process_finished",
}


class LifecycleTimingError(RuntimeError):
    """Raised when lifecycle markers cannot produce valid timing scopes."""


@dataclass(frozen=True)
class FrameLifecycleMarkers:
    """Monotonic timestamps emitted by a measured frame producer."""

    first_frame_completed_ns: int
    last_frame_completed_ns: int
    processed_frames: int
    instrumentation: str
    phase_completed_ns: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.first_frame_completed_ns <= 0:
            raise LifecycleTimingError("First-frame timestamp must be positive")
        if self.last_frame_completed_ns < self.first_frame_completed_ns:
            raise LifecycleTimingError("Last-frame timestamp precedes first frame")
        if self.processed_frames <= 0:
            raise LifecycleTimingError("Processed frame count must be positive")
        if not self.instrumentation:
            raise LifecycleTimingError("Lifecycle instrumentation must be identified")
        for name, completed_ns in self.phase_completed_ns.items():
            if not name or name in _RESERVED_CHECKPOINTS:
                raise LifecycleTimingError(f"Invalid lifecycle phase name: {name!r}")
            if (
                not isinstance(completed_ns, int)
                or isinstance(completed_ns, bool)
                or completed_ns <= 0
            ):
                raise LifecycleTimingError(
                    f"Lifecycle phase {name!r} must have a positive integer timestamp"
                )


def write_frame_markers(path: Path, markers: FrameLifecycleMarkers) -> None:
    """Write child-process frame boundaries for the parent benchmark runner."""
    markers.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "clock": _CLOCK,
        **asdict(markers),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_frame_markers(path: Path) -> FrameLifecycleMarkers:
    """Load and validate frame boundaries emitted by a measured subprocess."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleTimingError(f"Cannot read lifecycle markers {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LifecycleTimingError("Lifecycle marker document must be a JSON object")
    if payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise LifecycleTimingError("Unsupported lifecycle marker schema")
    if payload.get("clock") != _CLOCK:
        raise LifecycleTimingError("Lifecycle marker clock does not match runner clock")
    try:
        phase_payload = payload.get("phase_completed_ns", {})
        if not isinstance(phase_payload, dict):
            raise TypeError("phase_completed_ns must be an object")
        markers = FrameLifecycleMarkers(
            first_frame_completed_ns=int(payload["first_frame_completed_ns"]),
            last_frame_completed_ns=int(payload["last_frame_completed_ns"]),
            processed_frames=int(payload["processed_frames"]),
            instrumentation=str(payload["instrumentation"]),
            phase_completed_ns={
                str(name): int(completed_ns) for name, completed_ns in phase_payload.items()
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleTimingError("Lifecycle marker document is incomplete") from exc
    markers.validate()
    return markers


def _detailed_phase_summary(
    *,
    process_started_ns: int,
    process_finished_ns: int,
    markers: FrameLifecycleMarkers,
) -> dict[str, Any] | None:
    if not markers.phase_completed_ns:
        return None

    checkpoints = {
        "process_started": process_started_ns,
        **markers.phase_completed_ns,
        "first_frame_completed": markers.first_frame_completed_ns,
        "last_frame_completed": markers.last_frame_completed_ns,
        "process_finished": process_finished_ns,
    }
    for name, completed_ns in checkpoints.items():
        if completed_ns < process_started_ns or completed_ns > process_finished_ns:
            raise LifecycleTimingError(
                f"Lifecycle checkpoint {name!r} falls outside the measured process"
            )

    ordered = sorted(checkpoints.items(), key=lambda item: (item[1], item[0]))
    intervals_sec = {
        f"{previous_name}_to_{name}": (completed_ns - previous_ns) / 1_000_000_000
        for (previous_name, previous_ns), (name, completed_ns) in zip(
            ordered[:-1],
            ordered[1:],
            strict=True,
        )
    }
    return {
        "checkpoints_from_process_start_sec": {
            name: (completed_ns - process_started_ns) / 1_000_000_000
            for name, completed_ns in ordered
        },
        "intervals_sec": intervals_sec,
    }


def median_detailed_phase_intervals(
    lifecycle_summaries: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate project-specific lifecycle intervals across measured runs."""
    summaries = list(lifecycle_summaries)
    detailed_reports = [summary.get("detailed") for summary in summaries]
    if not any(detailed_reports):
        return {}
    if not all(isinstance(report, dict) for report in detailed_reports):
        raise LifecycleTimingError("Detailed lifecycle phases are missing from some runs")

    interval_reports = [report.get("intervals_sec") for report in detailed_reports]
    if not all(isinstance(intervals, dict) for intervals in interval_reports):
        raise LifecycleTimingError("Detailed lifecycle interval data is incomplete")

    expected_names = set(interval_reports[0])
    if any(set(intervals) != expected_names for intervals in interval_reports[1:]):
        raise LifecycleTimingError("Detailed lifecycle phase names changed between runs")

    medians: dict[str, float] = {}
    for name in sorted(expected_names):
        values = [intervals[name] for intervals in interval_reports]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise LifecycleTimingError(f"Detailed lifecycle interval {name!r} is invalid")
        medians[name] = float(statistics.median(values))
    return medians


def summarize_lifecycle(
    *,
    process_started_ns: int,
    process_finished_ns: int,
    markers: FrameLifecycleMarkers,
    expected_frames: int,
) -> dict[str, Any]:
    """Split measured process wall time into three exhaustive lifecycle scopes."""
    markers.validate()
    if expected_frames <= 0:
        raise LifecycleTimingError("Expected frame count must be positive")
    if markers.processed_frames != expected_frames:
        raise LifecycleTimingError(
            "Lifecycle frame count mismatch: "
            f"expected {expected_frames}, got {markers.processed_frames}"
        )
    if process_finished_ns <= process_started_ns:
        raise LifecycleTimingError("Measured process duration must be positive")
    if markers.first_frame_completed_ns < process_started_ns:
        raise LifecycleTimingError("First-frame marker precedes measured process")
    if markers.last_frame_completed_ns > process_finished_ns:
        raise LifecycleTimingError("Last-frame marker follows measured process")

    startup_ns = markers.first_frame_completed_ns - process_started_ns
    steady_ns = markers.last_frame_completed_ns - markers.first_frame_completed_ns
    finalize_ns = process_finished_ns - markers.last_frame_completed_ns
    total_ns = process_finished_ns - process_started_ns
    summary = {
        "clock": _CLOCK,
        "boundary_contract": _BOUNDARY_CONTRACT,
        "instrumentation": markers.instrumentation,
        "startup_sec": startup_ns / 1_000_000_000,
        "steady_state_frame_loop_sec": steady_ns / 1_000_000_000,
        "finalize_mux_sec": finalize_ns / 1_000_000_000,
        "total_sec": total_ns / 1_000_000_000,
        "processed_frames": markers.processed_frames,
        "steady_state_frames": max(0, markers.processed_frames - 1),
    }
    detailed = _detailed_phase_summary(
        process_started_ns=process_started_ns,
        process_finished_ns=process_finished_ns,
        markers=markers,
    )
    if detailed is not None:
        summary["detailed"] = detailed
    return summary
