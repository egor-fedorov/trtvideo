from trtvideo.video.bitrate import (
    AUTO_BITRATE_COMPLEXITY_EXPONENT,
    auto_bitrate_from_source,
)


def test_auto_bitrate_scales_with_pixels_sublinearly() -> None:
    bitrate, pixel_ratio, fps_ratio = auto_bitrate_from_source(
        source_bitrate=4_000_000,
        input_w=1280,
        input_h=720,
        output_w=2560,
        output_h=1440,
        input_fps=60.0,
        output_fps=60.0,
    )

    assert pixel_ratio == 4.0
    assert fps_ratio == 1.0
    assert bitrate == int(4_000_000 * (4.0**AUTO_BITRATE_COMPLEXITY_EXPONENT))
    assert bitrate < 4_000_000 * 4


def test_auto_bitrate_accounts_for_fps_ratio() -> None:
    bitrate, pixel_ratio, fps_ratio = auto_bitrate_from_source(
        source_bitrate=10_000_000,
        input_w=1920,
        input_h=1080,
        output_w=1920,
        output_h=1080,
        input_fps=30.0,
        output_fps=60.0,
    )

    assert pixel_ratio == 1.0
    assert fps_ratio == 2.0
    assert bitrate == int(10_000_000 * (2.0**AUTO_BITRATE_COMPLEXITY_EXPONENT))


def test_auto_bitrate_falls_back_to_neutral_ratios_for_invalid_inputs() -> None:
    bitrate, pixel_ratio, fps_ratio = auto_bitrate_from_source(
        source_bitrate=5_000_000,
        input_w=0,
        input_h=720,
        output_w=1280,
        output_h=720,
        input_fps=0.0,
        output_fps=60.0,
    )

    assert (bitrate, pixel_ratio, fps_ratio) == (5_000_000, 1.0, 1.0)
