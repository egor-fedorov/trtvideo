import json
from dataclasses import dataclass
from typing import Any

import pytest

from ai_media.video import info as video_info


@dataclass
class FakeCompletedProcess:
    returncode: int
    stdout: str
    stderr: str = ""


def _ffprobe_payload(*, stream_bit_rate: str | None, format_bit_rate: str | None) -> str:
    stream = {
        "codec_type": "video",
        "width": 1280,
        "height": 720,
        "r_frame_rate": "60000/1001",
        "nb_frames": "120",
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }
    if stream_bit_rate is not None:
        stream["bit_rate"] = stream_bit_rate

    payload: dict[str, Any] = {"streams": [stream], "format": {}}
    if format_bit_rate is not None:
        payload["format"]["bit_rate"] = format_bit_rate
    return json.dumps(payload)


def test_parse_fps_fraction() -> None:
    assert video_info._parse_fps("60000/1001") == pytest.approx(59.94005994)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("N/A", None),
        ("0", None),
        ("-1", None),
        ("12345", 12345),
    ],
)
def test_parse_optional_int(value: object, expected: int | None) -> None:
    assert video_info._parse_optional_int(value) == expected


def test_get_video_info_prefers_video_stream_bitrate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> FakeCompletedProcess:
        return FakeCompletedProcess(
            returncode=0,
            stdout=_ffprobe_payload(stream_bit_rate="4100000", format_bit_rate="4500000"),
        )

    monkeypatch.setattr(video_info.subprocess, "run", fake_run)

    info = video_info.get_video_info("input.mp4")

    assert info.width == 1280
    assert info.height == 720
    assert info.fps == pytest.approx(59.94005994)
    assert info.nb_frames == 120
    assert info.video_bit_rate == 4_100_000
    assert info.container_bit_rate == 4_500_000


def test_get_video_info_keeps_container_bitrate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> FakeCompletedProcess:
        return FakeCompletedProcess(
            returncode=0,
            stdout=_ffprobe_payload(stream_bit_rate=None, format_bit_rate="4500000"),
        )

    monkeypatch.setattr(video_info.subprocess, "run", fake_run)

    info = video_info.get_video_info("input.mp4")

    assert info.video_bit_rate is None
    assert info.container_bit_rate == 4_500_000


def test_get_video_info_exits_when_ffprobe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> FakeCompletedProcess:
        return FakeCompletedProcess(returncode=1, stdout="", stderr="bad input")

    monkeypatch.setattr(video_info.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        video_info.get_video_info("missing.mp4")
