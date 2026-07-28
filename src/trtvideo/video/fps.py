"""Frame-rate parsing helpers."""

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
