from fractions import Fraction

import pytest

from ai_media.video.fps import (
    format_nvenc_fps,
    gop_size_for_one_second,
    parse_fps,
    parse_fps_fraction,
)


def test_parse_fps_fraction_from_ffprobe_ratio() -> None:
    assert parse_fps_fraction("60000/1001") == Fraction(60000, 1001)


def test_parse_fps_returns_float() -> None:
    assert parse_fps("30000/1001") == pytest.approx(29.97002997)


def test_format_nvenc_fps_keeps_integer_rate_without_decimal() -> None:
    assert format_nvenc_fps("30/1") == "30"


def test_format_nvenc_fps_does_not_round_fractional_rate_to_integer() -> None:
    assert format_nvenc_fps("30000/1001") == str(float(Fraction(30000, 1001)))


def test_format_nvenc_fps_rejects_zero_denominator_rate() -> None:
    with pytest.raises(ValueError, match="invalid frame rate"):
        format_nvenc_fps("0/0")


def test_gop_size_for_one_second_uses_frame_count_interval() -> None:
    assert gop_size_for_one_second("24/1") == 24
    assert gop_size_for_one_second("30000/1001") == 30
    assert gop_size_for_one_second("60000/1001") == 60


def test_gop_size_for_one_second_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="invalid frame rate"):
        gop_size_for_one_second("0/0")
