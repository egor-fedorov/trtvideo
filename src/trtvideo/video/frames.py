"""Generic frame iterator lifecycle helpers."""

from collections.abc import Generator, Iterable
from itertools import islice


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
