"""Decoder frame-lifetime helpers without runtime-specific imports."""

from collections.abc import Callable, Generator, Sequence


def iter_locked_decode_frames[FrameT](
    fetch_batch: Callable[[int], Sequence[FrameT]],
    *,
    batch_size: int,
    release_batch: Callable[[], None],
) -> Generator[FrameT, None, None]:
    """Yield frames while keeping each decoder batch locked until GPU work completes."""
    while True:
        frames = fetch_batch(batch_size)
        if not frames:
            return

        try:
            yield from frames
        finally:
            release_batch()
