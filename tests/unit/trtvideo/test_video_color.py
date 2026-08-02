import pytest

from trtvideo.video.color import ColorContractError, SdrColorContract
from trtvideo.video.metadata import VideoMetadata


def _video_info(**overrides) -> VideoMetadata:
    values = {
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "fps_str": "24/1",
        "nb_frames": 100,
        **overrides,
    }
    return VideoMetadata(**values)


def test_sdr_contract_normalizes_unknown_hd_metadata() -> None:
    contract = SdrColorContract.from_video_info(_video_info(color_range="unknown"))

    assert contract.color_range == "tv"
    assert contract.color_space == "bt709"
    assert contract.color_transfer == "bt709"
    assert contract.color_primaries == "bt709"
    assert contract.cvcuda_spec_name == "bt709"
    assert contract.limited_range


def test_sdr_contract_maps_sd_and_full_range() -> None:
    contract = SdrColorContract.from_video_info(
        _video_info(width=640, height=480, color_range="pc", color_space="bt470bg")
    )

    assert contract.cvcuda_spec_name == "bt601"
    assert not contract.limited_range
    assert contract.ffmpeg_args() == [
        "-color_range",
        "pc",
        "-colorspace",
        "bt470bg",
        "-color_trc",
        "smpte170m",
        "-color_primaries",
        "smpte170m",
    ]


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_sdr_contract_rejects_hdr_transfer(transfer: str) -> None:
    with pytest.raises(ColorContractError, match="HDR input is not supported"):
        SdrColorContract.from_video_info(_video_info(color_transfer=transfer))
