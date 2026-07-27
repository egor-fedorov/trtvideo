"""Frame-rate parsing and encoder formatting helpers."""

from fractions import Fraction


def parse_fps_fraction(value: str) -> Fraction:
    """Parse ffprobe-style frame rate strings such as ``30000/1001``."""
    if "/" in value:
        numerator, denominator = map(int, value.split("/", 1))
        if denominator == 0:
            return Fraction(0, 1)
        return Fraction(numerator, denominator)
    return Fraction(value)


def parse_fps(value: str) -> float:
    """Parse a frame-rate string into a float for reporting and arithmetic."""
    return float(parse_fps_fraction(value))


def format_nvenc_fps(value: str) -> str:
    """Return a PyNvVideoCodec ``fps`` parameter without integer rounding."""
    fps = parse_fps_fraction(value)
    if fps <= 0:
        raise ValueError(f"invalid frame rate: {value}")
    if fps.denominator == 1:
        return str(fps.numerator)
    return str(float(fps))


def gop_size_for_one_second(value: str) -> int:
    """Return a GOP/IDR interval of roughly one second, measured in frames."""
    fps = parse_fps_fraction(value)
    if fps <= 0:
        raise ValueError(f"invalid frame rate: {value}")
    return max(1, int(fps + Fraction(1, 2)))
