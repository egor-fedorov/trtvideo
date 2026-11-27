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


@pytest.mark.parametrize("matrix", ["smpte240m", "fcc", "ycgco", "bt2020c", "reserved", "bogus"])
def test_explicit_unsupported_matrix_never_falls_back(matrix):
    with pytest.raises(ColorContractError, match=f"color_space='{matrix}'"):
        SdrColorContract.from_video_info(_video_info(color_space=matrix))


@pytest.mark.parametrize("missing", [None, "", "unknown", "unspecified"])
@pytest.mark.parametrize("width,expected", [(720, "smpte170m"), (1280, "bt709")])
def test_only_unspecified_matrix_uses_raster_default(missing, width, expected):
    color = SdrColorContract.from_video_info(
        _video_info(width=width, height=480, color_space=missing)
    )
    assert color.color_space == expected


@pytest.mark.parametrize("field", ["color_range", "color_transfer", "color_primaries"])
def test_invalid_signalling_is_rejected_at_probe_time(field):
    with pytest.raises(ColorContractError, match=field):
        SdrColorContract.from_video_info(_video_info(**{field: "reserved"}))


@pytest.mark.parametrize(
    "name,spec,code",
    [
        ("bt709", "bt709", 1),
        ("bt470bg", "bt601", 5),
        ("smpte170m", "bt601", 6),
        ("bt2020nc", "bt2020", 9),
    ],
)
def test_conversion_and_signalling_agree(name, spec, code):
    color = SdrColorContract.from_video_info(_video_info(color_space=name))
    assert color.cvcuda_spec_name == spec
    for codec in ("h264", "hevc"):
        assert f"matrix_coefficients={code}:" in color.bitstream_args(codec)[1]
