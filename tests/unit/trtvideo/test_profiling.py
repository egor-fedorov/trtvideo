from __future__ import annotations

from trtvideo.diagnostics.profiling import ProfileCollector


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
