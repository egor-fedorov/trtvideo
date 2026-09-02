"""Human and machine-readable production pipeline reporting."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

PROCESS_RESULT_DOCUMENT_TYPE = "trtvideo-process-result"
PROCESS_RESULT_SCHEMA_VERSION = 1
PROGRESS_EVENT_DOCUMENT_TYPE = "trtvideo-progress-event"
PROGRESS_EVENT_SCHEMA_VERSION = 1


class ProcessReportingError(RuntimeError):
    """Raised when a requested process report cannot be written."""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One interval-based frame-loop progress observation."""

    processed_frames: int
    total_frames: int | None
    completion_ratio: float | None
    frame_loop_elapsed_sec: float
    last_frame_processing_sec: float
    window_frames: int
    window_elapsed_sec: float
    window_average_sec_per_frame: float
    window_fps: float
    frame_loop_eta_sec: float | None

    def as_json(self) -> dict[str, object]:
        """Return the versioned JSONL representation."""
        return {
            "document_type": PROGRESS_EVENT_DOCUMENT_TYPE,
            "schema_version": PROGRESS_EVENT_SCHEMA_VERSION,
            "event": "progress",
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class ProcessTimingSummary:
    """Stable timing summary shared by text and JSON reports."""

    processed_frames: int
    warmup_frames_excluded: int
    average_frame_processing_sec: float
    min_frame_processing_sec: float
    max_frame_processing_sec: float
    post_warmup_average_frame_processing_sec: float
    frame_loop_sec: float
    frame_loop_fps: float
    active_pipeline_sec: float
    active_pipeline_fps: float
    pipeline_wall_sec: float
    pipeline_wall_fps: float

    def as_json(self) -> dict[str, object]:
        """Return a JSON-compatible timing object."""
        return {
            "frame_processing": {
                "average_sec": self.average_frame_processing_sec,
                "min_sec": self.min_frame_processing_sec,
                "max_sec": self.max_frame_processing_sec,
                "post_warmup_average_sec": (self.post_warmup_average_frame_processing_sec),
                "warmup_frames_excluded": self.warmup_frames_excluded,
            },
            "frame_loop": {
                "elapsed_sec": self.frame_loop_sec,
                "fps": self.frame_loop_fps,
            },
            "active_pipeline": {
                "elapsed_sec": self.active_pipeline_sec,
                "fps": self.active_pipeline_fps,
            },
            "pipeline_wall": {
                "elapsed_sec": self.pipeline_wall_sec,
                "fps": self.pipeline_wall_fps,
            },
        }


def build_progress_event(
    *,
    processed_frames: int,
    total_frames: int | None,
    frame_loop_elapsed_sec: float,
    last_frame_processing_sec: float,
    window_frames: int,
    window_elapsed_sec: float,
) -> ProgressEvent:
    """Build wall-clock progress metrics without turning text into an API."""
    window_average = window_elapsed_sec / window_frames if window_frames else 0.0
    window_fps = 1.0 / window_average if window_average > 0 else 0.0
    if total_frames is not None and total_frames > 0:
        completion_ratio = min(processed_frames / total_frames, 1.0)
        frame_loop_eta_sec = max(total_frames - processed_frames, 0) * window_average
    else:
        completion_ratio = None
        frame_loop_eta_sec = None
    return ProgressEvent(
        processed_frames=processed_frames,
        total_frames=total_frames,
        completion_ratio=completion_ratio,
        frame_loop_elapsed_sec=frame_loop_elapsed_sec,
        last_frame_processing_sec=last_frame_processing_sec,
        window_frames=window_frames,
        window_elapsed_sec=window_elapsed_sec,
        window_average_sec_per_frame=window_average,
        window_fps=window_fps,
        frame_loop_eta_sec=frame_loop_eta_sec,
    )


def summarize_process_timing(
    frame_times: list[float],
    *,
    warmup_frames: int,
    frame_loop_sec: float,
    active_pipeline_sec: float,
    pipeline_wall_sec: float,
) -> ProcessTimingSummary:
    """Summarize frame-loop and wider pipeline timing scopes."""
    processed_frames = len(frame_times)
    if not frame_times:
        return ProcessTimingSummary(
            processed_frames=0,
            warmup_frames_excluded=0,
            average_frame_processing_sec=0.0,
            min_frame_processing_sec=0.0,
            max_frame_processing_sec=0.0,
            post_warmup_average_frame_processing_sec=0.0,
            frame_loop_sec=frame_loop_sec,
            frame_loop_fps=0.0,
            active_pipeline_sec=active_pipeline_sec,
            active_pipeline_fps=0.0,
            pipeline_wall_sec=pipeline_wall_sec,
            pipeline_wall_fps=0.0,
        )

    excluded = warmup_frames if processed_frames > warmup_frames else 0
    measured = frame_times[excluded:]
    average = sum(frame_times) / processed_frames
    measured_average = sum(measured) / len(measured)
    return ProcessTimingSummary(
        processed_frames=processed_frames,
        warmup_frames_excluded=excluded,
        average_frame_processing_sec=average,
        min_frame_processing_sec=min(frame_times),
        max_frame_processing_sec=max(frame_times),
        post_warmup_average_frame_processing_sec=measured_average,
        frame_loop_sec=frame_loop_sec,
        frame_loop_fps=processed_frames / frame_loop_sec if frame_loop_sec > 0 else 0.0,
        active_pipeline_sec=active_pipeline_sec,
        active_pipeline_fps=(
            processed_frames / active_pipeline_sec if active_pipeline_sec > 0 else 0.0
        ),
        pipeline_wall_sec=pipeline_wall_sec,
        pipeline_wall_fps=processed_frames / pipeline_wall_sec if pipeline_wall_sec > 0 else 0.0,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    rounded = max(int(round(seconds)), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def render_progress(event: ProgressEvent) -> str:
    """Render one concise interactive progress line."""
    if event.total_frames is None or event.completion_ratio is None:
        progress = f"[{event.processed_frames}]"
    else:
        progress = (
            f"[{event.processed_frames}/{event.total_frames} {event.completion_ratio * 100:.1f}%]"
        )
    return (
        f"{progress} window {event.window_fps:.2f} FPS | "
        f"ETA {_format_duration(event.frame_loop_eta_sec)} | "
        f"last frame body {event.last_frame_processing_sec:.2f}s"
    )


def write_json_document(
    destination: Path,
    payload: dict[str, object],
    *,
    stdout: TextIO | None = None,
) -> None:
    """Write one complete JSON document to a path or stdout."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination == Path("-"):
        stream = stdout if stdout is not None else sys.stdout
        stream.write(text)
        stream.flush()
        return
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ProcessReportingError(f"Cannot write result JSON: {destination}") from exc


class JsonLinesWriter:
    """Write optional progress events without owning stdout."""

    def __init__(self, destination: Path | None, *, stdout: TextIO | None = None):
        self._destination = destination
        self._stdout = stdout
        self._stream: TextIO | None = None
        self._owns_stream = False

    def __enter__(self) -> JsonLinesWriter:
        if self._destination is None:
            return self
        if self._destination == Path("-"):
            self._stream = self._stdout if self._stdout is not None else sys.stdout
            return self
        try:
            self._destination.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._destination.open("w", encoding="utf-8")
            self._owns_stream = True
        except OSError as exc:
            raise ProcessReportingError(f"Cannot open progress JSONL: {self._destination}") from exc
        return self

    @property
    def enabled(self) -> bool:
        """Return whether events have a configured destination."""
        return self._destination is not None

    def write(self, event: ProgressEvent) -> None:
        """Append one compact JSON event when the sink is enabled."""
        if self._stream is None:
            return
        try:
            self._stream.write(json.dumps(event.as_json(), sort_keys=True) + "\n")
            self._stream.flush()
        except OSError as exc:
            raise ProcessReportingError("Cannot write progress JSONL") from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._owns_stream and self._stream is not None:
            self._stream.close()
        self._stream = None
