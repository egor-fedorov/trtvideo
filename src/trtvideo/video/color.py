"""SDR color contract shared by the production video path."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from trtvideo.video.metadata import VideoMetadata

_UNKNOWN_VALUES = {None, "", "unknown", "unspecified"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


# Keep CV-CUDA conversion and H.273 bitstream signalling in lockstep.
_MATRICES = {
    "bt709": ("bt709", 1),
    "bt470bg": ("bt601", 5),
    "smpte170m": ("bt601", 6),
    "bt2020nc": ("bt2020", 9),
}
_PRIMARIES = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "film": 8,
    "bt2020": 9,
    "smpte428": 10,
    "smpte431": 11,
    "smpte432": 12,
    "ebu3213": 22,
    "jedec-p22": 22,
}
_TRANSFERS = {
    "bt709": 1,
    "gamma22": 4,
    "gamma28": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "linear": 8,
    "log": 9,
    "log_sqrt": 10,
    "iec61966-2-4": 11,
    "bt1361e": 12,
    "iec61966-2-1": 13,
    "bt2020-10": 14,
    "bt2020-12": 15,
    "smpte428": 17,
}


class ColorContractError(ValueError):
    """Raised when input color metadata is outside the supported contract."""


def _value_or_default(value: str | None, default: str) -> str:
    return value if value not in _UNKNOWN_VALUES and value is not None else default


@dataclass(frozen=True, slots=True)
class SdrColorContract:
    """Normalized color metadata and conversion policy for one SDR input."""

    color_range: str
    color_space: str
    color_transfer: str
    color_primaries: str

    def __post_init__(self) -> None:
        if self.color_transfer in _HDR_TRANSFERS:
            raise ColorContractError(
                f"HDR input is not supported: color_transfer={self.color_transfer}. "
                "Convert/tonemap to SDR first."
            )
        for field, value, supported in (
            ("color_space", self.color_space, tuple(_MATRICES)),
            ("color_range", self.color_range, ("pc", "tv")),
            ("color_transfer", self.color_transfer, tuple(_TRANSFERS)),
            ("color_primaries", self.color_primaries, tuple(_PRIMARIES)),
        ):
            if value not in supported:
                raise ColorContractError(
                    f"Unsupported SDR {field}={value!r}. Convert explicitly to SDR BT.709 first."
                )

    @classmethod
    def from_video_info(cls, info: VideoMetadata) -> SdrColorContract:
        default_space = "bt709" if info.width >= 1280 or info.height >= 720 else "smpte170m"
        return cls(
            color_range=_value_or_default(info.color_range, "tv"),
            color_space=_value_or_default(info.color_space, default_space),
            color_transfer=_value_or_default(info.color_transfer, default_space),
            color_primaries=_value_or_default(info.color_primaries, default_space),
        )

    @property
    def limited_range(self) -> bool:
        return self.color_range != "pc"

    @property
    def cvcuda_spec_name(self) -> str:
        return _MATRICES[self.color_space][0]

    def ffmpeg_args(self) -> list[str]:
        """Return explicit FFmpeg output color tags."""
        return [
            "-color_range",
            self.color_range,
            "-colorspace",
            self.color_space,
            "-color_trc",
            self.color_transfer,
            "-color_primaries",
            self.color_primaries,
        ]

    def bitstream_args(self, codec: str, *, sar: Fraction = Fraction(1)) -> list[str]:
        """Set VUI as well as container tags: HEVC readers may prefer the SPS."""
        values = (
            f"colour_primaries={_PRIMARIES[self.color_primaries]}:"
            f"transfer_characteristics={_TRANSFERS[self.color_transfer]}:"
            f"matrix_coefficients={_MATRICES[self.color_space][1]}:"
            f"video_full_range_flag={int(not self.limited_range)}"
        )
        if sar != 1:
            values += f":sample_aspect_ratio={sar.numerator}/{sar.denominator}"
        return ["-bsf:v", f"{codec}_metadata={values}"]
