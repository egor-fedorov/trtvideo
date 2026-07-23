"""Lifecycle timing boundaries shared by video benchmark runners."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_CLOCK = "time.perf_counter_ns"
_BOUNDARY_CONTRACT = (
    "process_start -> first_frame_completed -> last_frame_completed -> process_exit"
)


class LifecycleTimingError(RuntimeError):
    """Raised when lifecycle markers cannot produce valid timing scopes."""


@dataclass(frozen=True)
class FrameLifecycleMarkers:
    """Monotonic timestamps emitted by a measured frame producer."""

    first_frame_completed_ns: int
    last_frame_completed_ns: int
    processed_frames: int
    instrumentation: str

    def validate(self) -> None:
        if self.first_frame_completed_ns <= 0:
            raise LifecycleTimingError("First-frame timestamp must be positive")
        if self.last_frame_completed_ns < self.first_frame_completed_ns:
            raise LifecycleTimingError("Last-frame timestamp precedes first frame")
        if self.processed_frames <= 0:
            raise LifecycleTimingError("Processed frame count must be positive")
        if not self.instrumentation:
            raise LifecycleTimingError("Lifecycle instrumentation must be identified")


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
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise LifecycleTimingError("Unsupported lifecycle marker schema")
    if payload.get("clock") != _CLOCK:
        raise LifecycleTimingError("Lifecycle marker clock does not match runner clock")
    try:
        markers = FrameLifecycleMarkers(
            first_frame_completed_ns=int(payload["first_frame_completed_ns"]),
            last_frame_completed_ns=int(payload["last_frame_completed_ns"]),
            processed_frames=int(payload["processed_frames"]),
            instrumentation=str(payload["instrumentation"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleTimingError("Lifecycle marker document is incomplete") from exc
    markers.validate()
    return markers


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
    return {
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
