from fractions import Fraction

import pytest

from trtvideo.video.fps import parse_fps, parse_fps_fraction


def test_parse_fps_fraction_from_ffprobe_ratio() -> None:
    assert parse_fps_fraction("60000/1001") == Fraction(60000, 1001)


def test_parse_fps_returns_float() -> None:
    assert parse_fps("30000/1001") == pytest.approx(29.97002997)
