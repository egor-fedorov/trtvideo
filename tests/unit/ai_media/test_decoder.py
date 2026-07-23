from ai_media.video.decoder import iter_locked_decode_frames


def test_decode_batch_is_released_before_fetching_the_next_batch() -> None:
    events: list[str] = []
    batches = iter([[1, 2], [3], []])

    def fetch(batch_size: int) -> list[int]:
        events.append(f"fetch:{batch_size}")
        return next(batches)

    def release() -> None:
        events.append("release")

    frames = iter_locked_decode_frames(fetch, batch_size=2, release_batch=release)

    assert next(frames) == 1
    events.append("processed:1")
    assert next(frames) == 2
    events.append("processed:2")
    assert next(frames) == 3

    assert events == [
        "fetch:2",
        "processed:1",
        "processed:2",
        "release",
        "fetch:2",
    ]


def test_closing_decode_iterator_releases_current_batch() -> None:
    releases = 0

    def release() -> None:
        nonlocal releases
        releases += 1

    frames = iter_locked_decode_frames(
        lambda _batch_size: [1, 2],
        batch_size=2,
        release_batch=release,
    )

    assert next(frames) == 1
    frames.close()

    assert releases == 1
