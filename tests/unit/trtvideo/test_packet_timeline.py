import pytest

from trtvideo.video.output.preservation import MediaPreservationError
from trtvideo.video.output.timeline import PacketTimeline


def packet(slot):
    return {"data": bytes([slot]), "timestamp": slot, "picture_type": 3}


def test_presentation_clock_survives_delayed_and_flush_packets():
    written = []
    clock = PacketTimeline(lambda p, pts, dur: written.append((p["data"], pts, dur)))
    assert clock.submit(-5) == 0
    clock.receive([])
    assert clock.submit(0) == 1
    clock.receive([packet(0), packet(1)])
    assert written == [(b"\x00", 0, 5)]
    assert clock.submit(10) == 2
    clock.set_end(12)
    clock.receive([packet(2)])
    clock.check_complete()
    assert written == [(b"\x00", 0, 5), (b"\x01", 5, 10), (b"\x02", 15, 2)]


@pytest.mark.parametrize("bad", [None, -(1 << 63), 101, 100])
def test_unknown_or_non_increasing_pts_is_not_synthesized(bad):
    clock = PacketTimeline(lambda *args: None)
    clock.submit(101)
    with pytest.raises(MediaPreservationError, match="timestamp|PTS"):
        clock.submit(bad)


def test_reject_unmatched_or_missing_packets():
    clock = PacketTimeline(lambda *args: None)
    clock.submit(42)
    clock.set_end(44)
    with pytest.raises(MediaPreservationError, match="Unexpected NVENC slot"):
        clock.receive([packet(1)])
    with pytest.raises(MediaPreservationError, match="Incomplete NVENC output"):
        clock.check_complete()
