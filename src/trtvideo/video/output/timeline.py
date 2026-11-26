"""Match presentation timestamps to NVENC access units without buffering a movie."""

from collections import deque
from collections.abc import Callable, Iterable

from trtvideo.video.nvcodec.encoder import NvencPacket
from trtvideo.video.output.preservation import MediaPreservationError


class PacketTimeline:
    """Output B-frames are disabled: slots must drain in submission order."""

    def __init__(self, write: Callable[[NvencPacket, int, int], None]) -> None:
        self.write = write
        self.first: int | None = None
        self.last: int | None = None
        self.end: int | None = None
        self._pts: deque[int] = deque()
        self._packets: deque[NvencPacket] = deque()
        self.submitted = 0
        self.written = 0

    def _append(self, pts: int) -> None:
        if not isinstance(pts, int) or pts == -(1 << 63):
            raise MediaPreservationError("Missing presentation timestamp on decoded frame")
        if self.last is not None and pts <= self.last:
            raise MediaPreservationError(f"Non-increasing video PTS: {pts} after {self.last}")
        self._pts.append(pts)

    def submit(self, pts: int) -> int:
        self._append(pts)
        if self.first is None:
            self.first = pts
        self.last = pts
        slot = self.submitted
        self.submitted += 1
        self._drain()
        return slot

    def set_end(self, end: int) -> None:
        if self.first is None:
            raise MediaPreservationError("No timestamped video frames decoded")
        self._append(end)
        self.end = end
        self._drain()

    def receive(self, packets: Iterable[NvencPacket]) -> None:
        for packet in packets:
            slot = self.written + len(self._packets)
            if packet["timestamp"] != slot or slot >= self.submitted:
                raise MediaPreservationError(
                    f"Unexpected NVENC slot {packet['timestamp']}, expected {slot}"
                )
            self._packets.append(packet)
        self._drain()

    def _drain(self) -> None:
        while self._packets and len(self._pts) >= 2:
            pts = self._pts.popleft()
            assert self.first is not None
            self.write(self._packets.popleft(), pts - self.first, self._pts[0] - pts)
            self.written += 1

    def check_complete(self) -> None:
        if self.end is None or self.written != self.submitted:
            raise MediaPreservationError(
                f"Incomplete NVENC output: {self.written}/{self.submitted} frames"
            )
