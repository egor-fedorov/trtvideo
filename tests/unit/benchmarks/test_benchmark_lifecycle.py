from __future__ import annotations

from pathlib import Path

import pytest

from trtvideo.benchmarking.lifecycle import (
    FrameLifecycleMarkers,
    LifecycleTimingError,
    load_frame_markers,
    median_detailed_phase_intervals,
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


def test_lifecycle_reports_detailed_phase_intervals() -> None:
    detailed_markers = FrameLifecycleMarkers(
        first_frame_completed_ns=2_000_000_000,
        last_frame_completed_ns=11_000_000_000,
        processed_frames=10,
        instrumentation="test-frame-loop",
        phase_completed_ns={
            "pipeline_created": 1_100_000_000,
            "encoder_initialized": 1_800_000_000,
            "frame_loop_completed": 11_100_000_000,
            "mux_completed": 12_000_000_000,
        },
    )

    summary = summarize_lifecycle(
        process_started_ns=1_000_000_000,
        process_finished_ns=13_000_000_000,
        markers=detailed_markers,
        expected_frames=10,
    )

    detailed = summary["detailed"]
    assert detailed["checkpoints_from_process_start_sec"]["pipeline_created"] == 0.1
    assert (
        detailed["intervals_sec"]["pipeline_created_to_encoder_initialized"] == 0.7
    )
    assert detailed["intervals_sec"]["last_frame_completed_to_frame_loop_completed"] == 0.1
    assert detailed["intervals_sec"]["mux_completed_to_process_finished"] == 1.0


def test_lifecycle_rejects_detailed_phase_outside_process() -> None:
    invalid_markers = FrameLifecycleMarkers(
        first_frame_completed_ns=2_000_000_000,
        last_frame_completed_ns=11_000_000_000,
        processed_frames=10,
        instrumentation="test-frame-loop",
        phase_completed_ns={"pipeline_created": 500_000_000},
    )

    with pytest.raises(LifecycleTimingError, match="outside the measured process"):
        summarize_lifecycle(
            process_started_ns=1_000_000_000,
            process_finished_ns=13_000_000_000,
            markers=invalid_markers,
            expected_frames=10,
        )


def test_lifecycle_aggregates_detailed_interval_medians() -> None:
    summaries = [
        {"detailed": {"intervals_sec": {"runtime_to_decoder": 0.2}}},
        {"detailed": {"intervals_sec": {"runtime_to_decoder": 0.4}}},
        {"detailed": {"intervals_sec": {"runtime_to_decoder": 0.3}}},
    ]

    assert median_detailed_phase_intervals(summaries) == {
        "runtime_to_decoder": 0.3
    }


def test_lifecycle_rejects_mixed_detailed_instrumentation() -> None:
    with pytest.raises(LifecycleTimingError, match="missing from some runs"):
        median_detailed_phase_intervals(
            [
                {"detailed": {"intervals_sec": {"runtime_to_decoder": 0.2}}},
                {},
            ]
        )


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
