from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from trtvideo.pipelines.reporting import (
    PROCESS_RESULT_DOCUMENT_TYPE,
    PROCESS_RESULT_SCHEMA_VERSION,
    PROGRESS_EVENT_DOCUMENT_TYPE,
    PROGRESS_EVENT_SCHEMA_VERSION,
    JsonLinesWriter,
    build_progress_event,
    render_progress,
    summarize_process_timing,
    write_json_document,
)


def test_progress_event_reports_window_rate_percent_and_eta() -> None:
    event = build_progress_event(
        processed_frames=10,
        total_frames=120,
        frame_loop_elapsed_sec=13.0,
        last_frame_processing_sec=1.5,
        window_frames=2,
        window_elapsed_sec=2.5,
    )

    assert event.completion_ratio == pytest.approx(1 / 12)
    assert event.window_average_sec_per_frame == 1.25
    assert event.window_fps == 0.8
    assert event.frame_loop_eta_sec == 137.5
    assert render_progress(event) == (
        "[10/120 8.3%] window 0.80 FPS | ETA 2m 18s | last frame body 1.50s"
    )
    assert event.as_json()["document_type"] == PROGRESS_EVENT_DOCUMENT_TYPE
    assert event.as_json()["schema_version"] == PROGRESS_EVENT_SCHEMA_VERSION


def test_progress_event_supports_unknown_total() -> None:
    event = build_progress_event(
        processed_frames=5,
        total_frames=None,
        frame_loop_elapsed_sec=2.0,
        last_frame_processing_sec=0.25,
        window_frames=1,
        window_elapsed_sec=0.25,
    )

    assert event.completion_ratio is None
    assert event.frame_loop_eta_sec is None
    assert render_progress(event) == ("[5] window 4.00 FPS | ETA unknown | last frame body 0.25s")


def test_process_summary_uses_configured_warmup_and_distinct_wall_scopes() -> None:
    summary = summarize_process_timing(
        [2.0, 1.0, 1.0, 1.0],
        warmup_frames=2,
        frame_loop_sec=4.0,
        active_pipeline_sec=5.0,
        pipeline_wall_sec=6.0,
    )

    assert summary.processed_frames == 4
    assert summary.warmup_frames_excluded == 2
    assert summary.average_frame_processing_sec == 1.25
    assert summary.post_warmup_average_frame_processing_sec == 1.0
    assert summary.frame_loop_fps == 1.0
    assert summary.active_pipeline_fps == 0.8
    assert summary.pipeline_wall_fps == pytest.approx(2 / 3)
    assert summary.as_json() == {
        "frame_processing": {
            "average_sec": 1.25,
            "min_sec": 1.0,
            "max_sec": 2.0,
            "post_warmup_average_sec": 1.0,
            "warmup_frames_excluded": 2,
        },
        "frame_loop": {"elapsed_sec": 4.0, "fps": 1.0},
        "active_pipeline": {"elapsed_sec": 5.0, "fps": 0.8},
        "pipeline_wall": {"elapsed_sec": 6.0, "fps": pytest.approx(2 / 3)},
    }


def test_process_summary_keeps_all_frames_when_warmup_would_remove_everything() -> None:
    summary = summarize_process_timing(
        [0.5],
        warmup_frames=1,
        frame_loop_sec=0.5,
        active_pipeline_sec=0.6,
        pipeline_wall_sec=0.7,
    )

    assert summary.warmup_frames_excluded == 0
    assert summary.post_warmup_average_frame_processing_sec == 0.5


def test_json_document_can_use_stdout_without_human_text() -> None:
    stdout = io.StringIO()
    payload: dict[str, object] = {
        "document_type": PROCESS_RESULT_DOCUMENT_TYPE,
        "schema_version": PROCESS_RESULT_SCHEMA_VERSION,
        "status": "completed",
    }

    write_json_document(Path("-"), payload, stdout=stdout)

    assert json.loads(stdout.getvalue()) == payload


def test_json_document_creates_parent_directory(tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "result.json"

    write_json_document(destination, {"status": "completed"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "completed"}


def test_json_lines_writer_emits_one_compact_event_per_line(tmp_path: Path) -> None:
    destination = tmp_path / "progress" / "events.jsonl"
    event = build_progress_event(
        processed_frames=1,
        total_frames=2,
        frame_loop_elapsed_sec=0.5,
        last_frame_processing_sec=0.4,
        window_frames=1,
        window_elapsed_sec=0.5,
    )

    with JsonLinesWriter(destination) as writer:
        assert writer.enabled
        writer.write(event)
        writer.write(event)

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["event"] == "progress" for line in lines)


def test_disabled_json_lines_writer_is_a_noop() -> None:
    event = build_progress_event(
        processed_frames=1,
        total_frames=1,
        frame_loop_elapsed_sec=0.1,
        last_frame_processing_sec=0.1,
        window_frames=1,
        window_elapsed_sec=0.1,
    )

    with JsonLinesWriter(None) as writer:
        assert not writer.enabled
        writer.write(event)
