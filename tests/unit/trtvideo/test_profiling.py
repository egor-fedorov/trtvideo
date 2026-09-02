from __future__ import annotations

import io
import json
from pathlib import Path

from trtvideo.diagnostics.profiling import ProfileCollector, write_profile_report
from trtvideo.video.metadata import VideoMetadata


class FakeEvent:
    def __init__(self, timestamp_ms: float):
        self.timestamp_ms = timestamp_ms
        self.closed = False

    def elapsed_time(self, end_event: FakeEvent) -> float:
        return end_event.timestamp_ms - self.timestamp_ms

    def close(self) -> None:
        self.closed = True


def test_profile_collector_uses_runtime_synchronizer_and_caches_summary() -> None:
    synchronizations = 0

    def synchronize() -> None:
        nonlocal synchronizations
        synchronizations += 1

    collector = ProfileCollector(
        ["preprocess", "inference"],
        gpu_stages=["preprocess", "inference"],
        synchronize=synchronize,
        skip_warmup=1,
    )
    events = (
        FakeEvent(0.0),
        FakeEvent(2.0),
        FakeEvent(5.0),
        FakeEvent(10.0),
        FakeEvent(11.5),
        FakeEvent(15.5),
    )
    collector.commit(events[:3])
    collector.commit(events[3:])

    summary = collector.summary([0.010, 0.020])
    cached = collector.summary([0.010, 0.020])

    assert summary["warmup_frames"] == 1
    assert summary["stage_ms"] == {"preprocess": 1.5, "inference": 4.0}
    assert summary["processing_fps"] == 50.0
    assert cached is summary
    assert synchronizations == 1
    assert all(event.closed for event in events)


class FakeRuntime:
    gpu_name = "Fake GPU"
    input_w = 1280
    input_h = 720
    output_w = 2560
    output_h = 1440

    def synchronize(self) -> None:
        pass

    def peak_memory_allocated_mb(self) -> float:
        return 123.5


def test_profile_report_normalizes_stage_names(tmp_path: Path) -> None:
    collector = ProfileCollector(
        ["TRT inference"],
        gpu_stages=["TRT inference"],
        synchronize=lambda: None,
        skip_warmup=0,
    )
    collector.commit((FakeEvent(1.0), FakeEvent(3.0)))
    output_path = tmp_path / "profile.json"

    write_profile_report(
        output_path,
        collector=collector,
        runtime=FakeRuntime(),
        video_info=VideoMetadata(1280, 720, 24.0, "24/1", 1),
        engine_path=Path("model.engine"),
        input_path=Path("input.mp4"),
        media_output_path=Path("output.mp4"),
        frame_times=[0.01],
        wall_total_sec=0.02,
        stage_key_map={"TRT inference": "trt"},
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["stage_ms"] == {"trt": 2.0}
    assert report["throughput_fps"] == 50.0
    assert report["gpu_peak_mem_mb"] == 123.5


def test_profile_table_uses_the_requested_human_output_stream() -> None:
    collector = ProfileCollector(
        ["TRT inference"],
        gpu_stages=["TRT inference"],
        synchronize=lambda: None,
        skip_warmup=0,
    )
    collector.commit((FakeEvent(1.0), FakeEvent(3.0)))
    output = io.StringIO()

    collector.print_table(1280, 720, 2560, 1440, [0.01], output=output)

    assert "Profiling: 1 frames" in output.getvalue()
    assert "TRT inference" in output.getvalue()
