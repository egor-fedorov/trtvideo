"""CUDA event-based profiling and reports for pipeline stages."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from trtvideo.runtime import RuntimeEngine
from trtvideo.video.metadata import VideoMetadata


class ProfileCollector:
    """Collects CUDA event timings for named pipeline stages.

    GPU stages are measured via sequential CUDA events (N+1 events
    for N stages). CPU stages use wall-clock time.perf_counter().

    Usage:
        collector = ProfileCollector(
            ["decode", "preprocess", "trt", "postprocess", "encode"],
            gpu_stages=["preprocess", "trt", "postprocess"],
        )
        # In the hot path - only a tuple of runtime-specific CUDA events:
        e0, e1, e2, e3 = (runtime.create_timing_event() for _ in range(4))
        e0.record(stream); ...; e1.record(stream); ...; e3.record(stream)
        collector.commit((e0, e1, e2, e3))
    """

    def __init__(
        self,
        stage_names: list[str],
        gpu_stages: list[str],
        synchronize: Callable[[], None],
        skip_warmup: int = 1,
    ):
        self.stage_names = stage_names
        self.gpu_stages = gpu_stages
        self._synchronize = synchronize
        self.skip_warmup = skip_warmup
        self._events: list[tuple] = []
        self._wall_times: dict[str, list[float]] = {
            name: [] for name in stage_names if name not in gpu_stages
        }
        self._summary_cache: dict[str, Any] | None = None

    def record_wall_time(self, stage_name: str, duration_s: float) -> None:
        """Record a wall-clock measurement for a CPU-bound stage (in seconds)."""
        self._wall_times[stage_name].append(duration_s)
        self._summary_cache = None

    def commit(self, events: tuple) -> None:
        """Store sequential CUDA events for one frame."""
        self._events.append(events)
        self._summary_cache = None

    @property
    def committed_count(self) -> int:
        return len(self._events)

    def summary(self, frame_times: list[float]) -> dict[str, Any]:
        """Return machine-readable profiling summary."""
        if self._summary_cache is not None:
            return self._summary_cache

        self._synchronize()

        n_total = len(self._events)
        skip = self.skip_warmup if n_total > self.skip_warmup else 0
        events = self._events[skip:]
        n = len(events)

        stage_ms: dict[str, float] = {}
        if n > 0:
            for name in self.stage_names:
                if name in self.gpu_stages:
                    idx = self.gpu_stages.index(name)
                    total_ms = sum(ev[idx].elapsed_time(ev[idx + 1]) for ev in events)
                    stage_ms[name] = total_ms / n
                else:
                    wall_vals = self._wall_times.get(name, [])
                    if len(wall_vals) > skip:
                        vals = wall_vals[skip:]
                        stage_ms[name] = sum(vals) / len(vals) * 1000

        measured_frame_times = frame_times[skip:]
        wall_avg_sec = (
            sum(measured_frame_times) / len(measured_frame_times) if measured_frame_times else 0.0
        )
        min_frame_sec = min(measured_frame_times) if measured_frame_times else 0.0
        max_frame_sec = max(measured_frame_times) if measured_frame_times else 0.0
        processing_fps = 1.0 / wall_avg_sec if wall_avg_sec > 0 else 0.0
        result = {
            "warmup_frames": skip,
            "frames": len(measured_frame_times),
            "processing_fps": processing_fps,
            "avg_frame_sec": wall_avg_sec,
            "avg_frame_ms": wall_avg_sec * 1000,
            "min_frame_ms": min_frame_sec * 1000,
            "max_frame_ms": max_frame_sec * 1000,
            "stage_ms": stage_ms,
        }
        for frame_events in self._events:
            for event in frame_events:
                close = getattr(event, "close", None)
                if close is not None:
                    close()
        self._events.clear()
        self._summary_cache = result
        return result

    def print_table(
        self,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        frame_times: list[float],
        output: TextIO | None = None,
    ) -> None:
        """Print the profiling table."""
        summary = self.summary(frame_times)
        n = int(summary["frames"])
        if n == 0:
            return

        stage_ms = summary["stage_ms"]
        rows = [
            (name if name in self.gpu_stages else name + " *", stage_ms[name])
            for name in self.stage_names
            if name in stage_ms
        ]

        total_avg = sum(ms for _, ms in rows)
        stream = output if output is not None else sys.stderr

        sep = "=" * 65
        dash = "-" * 65
        print(f"\n{sep}", file=stream)
        print(
            f"Profiling: {n} frames, {in_w}x{in_h} \u2192 {out_w}x{out_h}",
            file=stream,
        )
        print(sep, file=stream)
        print(f"{'Stage':<40s} {'Average':>8s} {'Share':>8s}", file=stream)
        print(dash, file=stream)
        for label, ms in rows:
            pct = ms / total_avg * 100 if total_avg > 0 else 0
            print(f"{label:<40s} {ms:>7.1f}ms {pct:>7.1f}%", file=stream)
        print(dash, file=stream)
        print(f"{'TOTAL':<40s} {total_avg:>7.1f}ms {'100.0%':>8s}", file=stream)

        print(f"{'FPS (processing)':<40s} {summary['processing_fps']:>7.1f}", file=stream)
        print(sep, file=stream)


def write_profile_report(
    output_path: Path,
    *,
    collector: ProfileCollector,
    runtime: RuntimeEngine,
    video_info: VideoMetadata,
    engine_path: Path,
    input_path: Path,
    media_output_path: Path,
    frame_times: list[float],
    wall_total_sec: float,
    stage_key_map: dict[str, str],
) -> None:
    """Write the machine-readable report for an isolated stage profile."""
    profile = collector.summary(frame_times)
    stage_ms = profile.get("stage_ms", {})
    normalized_stage_ms = {stage_key_map.get(name, name): value for name, value in stage_ms.items()}
    report = {
        "engine": str(engine_path),
        "gpu": runtime.gpu_name,
        "input": str(input_path),
        "output": str(media_output_path),
        "input_resolution": f"{video_info.width}x{video_info.height}",
        "output_resolution": f"{runtime.output_w}x{runtime.output_h}",
        "processed_frames": len(frame_times),
        "frames": profile.get("frames", len(frame_times)),
        "warmup_frames": profile.get("warmup_frames", 0),
        "processing_fps": profile.get("processing_fps", 0.0),
        "throughput_fps": (len(frame_times) / wall_total_sec if wall_total_sec > 0 else 0.0),
        "avg_frame_sec": profile.get("avg_frame_sec", 0.0),
        "avg_frame_ms": profile.get("avg_frame_ms", 0.0),
        "min_frame_ms": profile.get("min_frame_ms", 0.0),
        "max_frame_ms": profile.get("max_frame_ms", 0.0),
        "wall_total_sec": wall_total_sec,
        "stage_ms": normalized_stage_ms,
        "gpu_peak_mem_mb": runtime.peak_memory_allocated_mb(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
