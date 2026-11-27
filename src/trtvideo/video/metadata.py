"""Normalized video metadata used by processing components."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from trtvideo.video.geometry import DisplayGeometry


@dataclass(frozen=True, slots=True)
class VideoTrackProperties:
    """Descriptive properties of the selected track, not its encoded bitstream."""

    language: str | None = None
    title: str | None = None
    handler_name: str | None = None
    dispositions: frozenset[str] = frozenset()

    @classmethod
    def from_stream(cls, stream: dict[str, Any]) -> "VideoTrackProperties":
        tags = {key.lower(): value for key, value in stream.get("tags", {}).items()}
        language = tags.get("language")
        handler = tags.get("handler_name")
        dispositions = stream.get("disposition", {})
        return cls(
            language=language if language and language != "und" else None,
            title=tags.get("title") or None,
            handler_name=handler if handler and handler != "VideoHandler" else None,
            dispositions=frozenset(
                name
                for name in (
                    "default",
                    "original",
                    "dub",
                    "comment",
                    "forced",
                    "hearing_impaired",
                    "visual_impaired",
                )
                if dispositions.get(name) == 1
            ),
        )


EMPTY_TRACK_PROPERTIES = VideoTrackProperties()


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Primary video-stream metadata required by the production pipeline."""

    width: int
    height: int
    fps: float
    fps_str: str
    nb_frames: int
    pix_fmt: str | None = None
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    video_bit_rate: int | None = None
    container_bit_rate: int | None = None
    time_base: Fraction | None = None
    start_pts: int | None = None
    duration_ts: int | None = None
    geometry: DisplayGeometry = DisplayGeometry()
    track_properties: VideoTrackProperties = EMPTY_TRACK_PROPERTIES
