"""Bitrate estimation helpers shared by video backends and tests."""

AUTO_BITRATE_COMPLEXITY_EXPONENT = 0.6


def crf_to_bitrate(crf: int, width: int, height: int, fps: float) -> int:
    """Estimate bitrate from a CRF-like value.

    bpp ~ 0.1 at CRF 23, doubles every 6 units.
    """
    bpp = 0.1 * (2.0 ** ((23 - crf) / 6.0))
    return int(bpp * width * height * fps)


def auto_bitrate_from_source(
    *,
    source_bitrate: int,
    input_w: int,
    input_h: int,
    output_w: int,
    output_h: int,
    input_fps: float,
    output_fps: float,
) -> tuple[int, float, float]:
    """Estimate target bitrate from source bitrate and output complexity."""
    input_pixels = input_w * input_h
    output_pixels = output_w * output_h
    pixel_ratio = output_pixels / input_pixels if input_pixels > 0 else 1.0
    fps_ratio = output_fps / input_fps if input_fps > 0 and output_fps > 0 else 1.0
    scale = (pixel_ratio * fps_ratio) ** AUTO_BITRATE_COMPLEXITY_EXPONENT
    return int(source_bitrate * scale), pixel_ratio, fps_ratio
