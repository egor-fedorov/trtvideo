from __future__ import annotations

from pathlib import Path

import pytest

from trtvideo.benchmarking.lifecycle import (
    FrameLifecycleMarkers,
    LifecycleTimingError,
    load_frame_markers,
    summarize_lifecycle,
    write_frame_markers,
)


def markers() -> FrameLifecycleMarkers:
    return FrameLifecycleMarkers(
        first_frame_completed_ns=2_000_000_000,
        last_frame_completed_ns=11_000_000_000,
        processed_frames=10,
        instrumentation="test-frame-loop",
    )


def test_lifecycle_scopes_exhaust_process_wall_time() -> None:
    summary = summarize_lifecycle(
        process_started_ns=1_000_000_000,
        process_finished_ns=13_000_000_000,
        markers=markers(),
        expected_frames=10,
    )

    assert summary["startup_sec"] == 1.0
    assert summary["steady_state_frame_loop_sec"] == 9.0
    assert summary["finalize_mux_sec"] == 2.0
    assert summary["total_sec"] == 12.0
    assert summary["steady_state_frames"] == 9


def test_lifecycle_markers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.json"

    write_frame_markers(path, markers())

    assert load_frame_markers(path) == markers()


def test_lifecycle_rejects_marker_outside_process() -> None:
    with pytest.raises(LifecycleTimingError, match="precedes measured process"):
        summarize_lifecycle(
            process_started_ns=3_000_000_000,
            process_finished_ns=13_000_000_000,
            markers=markers(),
            expected_frames=10,
        )


def test_lifecycle_rejects_frame_count_mismatch() -> None:
    with pytest.raises(LifecycleTimingError, match="frame count mismatch"):
        summarize_lifecycle(
            process_started_ns=1_000_000_000,
            process_finished_ns=13_000_000_000,
            markers=markers(),
            expected_frames=11,
        )
