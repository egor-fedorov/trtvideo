"""Decoder frame-lifetime helpers without runtime-specific imports."""

from collections.abc import Callable, Generator, Iterable, Sequence
from itertools import islice


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


def iter_limited_frames[FrameT](
    frames: Iterable[FrameT],
    *,
    limit: int,
) -> Generator[FrameT, None, None]:
    """Yield at most ``limit`` frames and close an owned source iterator."""
    iterator = iter(frames)
    limited = islice(iterator, limit) if limit > 0 else iterator
    try:
        yield from limited
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
