"""CUDA event-based profiling for pipeline stages."""

import torch


class ProfileCollector:
    """Collects CUDA event timings for named pipeline stages.

    GPU stages are measured via sequential CUDA events (N+1 events
    for N stages). CPU stages use wall-clock time.perf_counter().

    Usage:
        collector = ProfileCollector(
            ["decode", "preprocess", "trt", "postprocess", "encode"],
            gpu_stages=["preprocess", "trt", "postprocess"],
        )
        # In the hot path — only tuple of events:
        e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        e0.record(stream); ...; e1.record(stream); ...; e3.record(stream)
        collector.commit((e0, e1, e2, e3))
    """

    def __init__(
        self,
        stage_names: list[str],
        gpu_stages: list[str],
        skip_warmup: int = 1,
    ):
        self.stage_names = stage_names
        self.gpu_stages = gpu_stages
        self.skip_warmup = skip_warmup
        self._events: list[tuple] = []
        self._wall_times: dict[str, list[float]] = {
            name: [] for name in stage_names if name not in gpu_stages
        }

    def record_wall_time(self, stage_name: str, duration_s: float) -> None:
        """Record a wall-clock measurement for a CPU-bound stage (in seconds)."""
        self._wall_times[stage_name].append(duration_s)

    def commit(self, events: tuple) -> None:
        """Store sequential CUDA events for one frame."""
        self._events.append(events)

    @property
    def committed_count(self) -> int:
        return len(self._events)

    def print_table(
        self,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        frame_times: list[float],
    ) -> None:
        """Print the profiling table."""
        torch.cuda.synchronize()

        n_total = len(self._events)
        skip = self.skip_warmup if n_total > self.skip_warmup else 0
        events = self._events[skip:]
        n = len(events)
        if n == 0:
            return

        rows: list[tuple[str, float]] = []
        for name in self.stage_names:
            if name in self.gpu_stages:
                idx = self.gpu_stages.index(name)
                total_ms = sum(ev[idx].elapsed_time(ev[idx + 1]) for ev in events)
                rows.append((name, total_ms / n))
            else:
                wall_vals = self._wall_times.get(name, [])
                if len(wall_vals) > skip:
                    vals = wall_vals[skip:]
                    rows.append((name + " *", sum(vals) / len(vals) * 1000))

        total_avg = sum(ms for _, ms in rows)

        sep = "=" * 65
        dash = "-" * 65
        print(f"\n{sep}")
        print(f"Profiling: {n} frames, {in_w}x{in_h} \u2192 {out_w}x{out_h}")
        print(sep)
        print(f"{'Stage':<40s} {'Average':>8s} {'Share':>8s}")
        print(dash)
        for label, ms in rows:
            pct = ms / total_avg * 100 if total_avg > 0 else 0
            print(f"{label:<40s} {ms:>7.1f}ms {pct:>7.1f}%")
        print(dash)
        print(f"{'TOTAL':<40s} {total_avg:>7.1f}ms {'100.0%':>8s}")

        # Wall-clock FPS
        ft = frame_times[skip:]
        if ft:
            wall_avg = sum(ft) / len(ft)
            wall_fps = 1.0 / wall_avg if wall_avg > 0 else 0
            print(f"{'FPS (wall-clock)':<40s} {wall_fps:>7.1f}")
        print(sep)
