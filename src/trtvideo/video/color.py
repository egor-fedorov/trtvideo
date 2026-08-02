"""SDR color contract shared by the production video path."""

from __future__ import annotations

from dataclasses import dataclass

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
