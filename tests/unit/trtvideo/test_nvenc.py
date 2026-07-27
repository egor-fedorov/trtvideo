from __future__ import annotations

import pytest

from trtvideo.video.nvenc import NvencCbrContract


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
