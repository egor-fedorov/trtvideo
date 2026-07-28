from __future__ import annotations

from fractions import Fraction

import pytest

from trtvideo.video.nvcodec.encoder import (
    NvencCbrContract,
    format_nvenc_fps,
    gop_size_for_one_second,
)


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


def test_nvenc_cbr_contract_matches_pynvcodec_and_ffmpeg() -> None:
    contract = NvencCbrContract(bitrate_bps=60_000_000, gop_frames=24)

    pynvcodec = contract.pynvcodec_options()
    ffmpeg = contract.ffmpeg_options()

    assert pynvcodec["rc"] == "cbr"
    assert pynvcodec["maxbitrate"] == 60_000_000
    assert pynvcodec["vbvbufsize"] == 120_000_000
    assert pynvcodec["vbvinit"] == 60_000_000
    assert pynvcodec["bf"] == 0
    assert ffmpeg[ffmpeg.index("-rc") + 1] == "cbr"
    assert ffmpeg[ffmpeg.index("-maxrate") + 1] == "60000000"
    assert ffmpeg[ffmpeg.index("-bufsize") + 1] == "120000000"
    assert ffmpeg[ffmpeg.index("-rc_init_occupancy") + 1] == "60000000"
    assert ffmpeg[ffmpeg.index("-multipass") + 1] == "disabled"


def test_nvenc_cbr_contract_serializes_all_parity_fields() -> None:
    values = NvencCbrContract(
        bitrate_bps=35_000_000,
        gop_frames=30,
    ).as_dict()

    assert values == {
        "codec": "h264",
        "preset": "p4",
        "tuning": "high_quality",
        "rate_control": "cbr",
        "target_bitrate_bps": 35_000_000,
        "min_bitrate_bps": 35_000_000,
        "max_bitrate_bps": 35_000_000,
        "vbv_buffer_bits": 70_000_000,
        "vbv_initial_delay_bits": 35_000_000,
        "multipass": "disabled",
        "lookahead_frames": 0,
        "spatial_aq": False,
        "temporal_aq": False,
        "gop_frames": 30,
        "b_frames": 0,
    }


@pytest.mark.parametrize(
    ("bitrate", "gop", "codec"),
    [(0, 24, "h264"), (1, 0, "h264"), (1, 24, "av1")],
)
def test_nvenc_cbr_contract_rejects_invalid_values(
    bitrate: int,
    gop: int,
    codec: str,
) -> None:
    with pytest.raises(ValueError):
        NvencCbrContract(bitrate_bps=bitrate, gop_frames=gop, codec=codec)
