"""Normalized video metadata used by processing components."""

from dataclasses import dataclass
from fractions import Fraction

from trtvideo.video.geometry import DisplayGeometry


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
