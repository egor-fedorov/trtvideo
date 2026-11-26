"""Display geometry of an unrotated, coded video raster."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

_ONE = 1 << 16
_QUARTER_TURNS = {(1, 0, 0, 1): 0, (0, -1, 1, 0): 90, (-1, 0, 0, -1): 180, (0, 1, -1, 0): 270}


def _angle(value: object) -> int:
    angle = Fraction(str(value)) % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError(f"Unsupported video display rotation: {value}")
    return int(angle)


def _matrix_rotation(text: str, width: int, height: int) -> int:
    try:
        rows = [line.split(":")[1].split() for line in text.splitlines() if line.strip()]
        (a, b, u), (c, d, v), (x, y, w) = [[int(i) for i in row] for row in rows]
    except (IndexError, ValueError) as exc:
        raise ValueError("Malformed video display matrix") from exc
    # Accept a pure quarter turn, optionally translated back to the raster origin.
    for linear, angle in _QUARTER_TURNS.items():
        if (a, b, c, d) == tuple(k * _ONE for k in linear):
            origin = (-min(0, a * width, c * height), -min(0, b * width, d * height))
            if (u, v, w) == (0, 0, 1 << 30) and (x, y) in ((0, 0), origin):
                return angle
    raise ValueError("Unsupported video display matrix (reflection, scaling or non-quarter turn)")


def _ratio(value: object) -> Fraction | None:
    if value in (None, "", "N/A", "0:1"):
        return None
    try:
        ratio = Fraction(str(value).replace(":", "/"))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid video aspect ratio: {value}") from exc
    if ratio <= 0:
        raise ValueError(f"Invalid video aspect ratio: {value}")
    return ratio


@dataclass(frozen=True, slots=True)
class DisplayGeometry:
    rotation: int = 0
    sar: Fraction = Fraction(1)

    @classmethod
    def from_stream(cls, stream: dict[str, Any]) -> "DisplayGeometry":
        width, height = int(stream["width"]), int(stream["height"])
        sar = _ratio(stream.get("sample_aspect_ratio"))
        dar = _ratio(stream.get("display_aspect_ratio"))
        coded_ratio = Fraction(width, height)
        if sar is not None and dar is not None and sar * coded_ratio != dar:
            raise ValueError("Conflicting sample and display aspect ratios")
        sar = sar or (dar / coded_ratio if dar else Fraction(1))
        if max(sar.numerator, sar.denominator) > 65535:
            raise ValueError("Sample aspect ratio cannot be represented in H.264/HEVC")
        angles = {
            _angle(value)
            for key, value in stream.get("tags", {}).items()
            if key.lower() == "rotate"
        }
        for item in stream.get("side_data_list", []):
            if item.get("side_data_type") == "Display Matrix":
                angles.add(_matrix_rotation(item.get("displaymatrix", ""), width, height))
                if "rotation" in item:
                    angles.add(_angle(item["rotation"]))
        if len(angles) > 1:
            raise ValueError("Conflicting video rotation tag and display matrix")
        return cls(next(iter(angles), 0), sar)


DEFAULT_GEOMETRY = DisplayGeometry()
