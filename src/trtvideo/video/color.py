"""SDR color contract shared by the production video path."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from trtvideo.video.metadata import VideoMetadata

_UNKNOWN_VALUES = {None, "", "unknown", "reserved"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


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

    @classmethod
    def from_video_info(cls, info: VideoMetadata) -> SdrColorContract:
        if info.color_transfer in _HDR_TRANSFERS:
            raise ColorContractError(
                "HDR input is not supported by the current SDR RGB model contract: "
                f"color_transfer={info.color_transfer}. Convert/tonemap to SDR first."
            )

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
        if self.color_space in {"bt2020nc", "bt2020c"}:
            return "bt2020"
        if self.color_space in {"smpte170m", "bt470bg", "bt470m"}:
            return "bt601"
        return "bt709"

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
        # H.273 code points used by FFmpeg's h264_metadata and hevc_metadata.
        primaries = {
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
        transfers = {
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
        matrices = {
            "gbr": 0,
            "bt709": 1,
            "fcc": 4,
            "bt470bg": 5,
            "smpte170m": 6,
            "smpte240m": 7,
            "ycgco": 8,
            "bt2020nc": 9,
            "bt2020c": 10,
            "smpte2085": 11,
            "chroma-derived-nc": 12,
            "chroma-derived-c": 13,
            "ictcp": 14,
            "ipt-c2": 15,
            "ycgco-re": 16,
            "ycgco-ro": 17,
        }
        try:
            values = (
                f"colour_primaries={primaries[self.color_primaries]}:"
                f"transfer_characteristics={transfers[self.color_transfer]}:"
                f"matrix_coefficients={matrices[self.color_space]}:"
                f"video_full_range_flag={int(not self.limited_range)}"
            )
        except KeyError as exc:
            raise ColorContractError(f"Cannot signal SDR color metadata: {exc}") from exc
        if sar != 1:
            values += f":sample_aspect_ratio={sar.numerator}/{sar.denominator}"
        return ["-bsf:v", f"{codec}_metadata={values}"]
