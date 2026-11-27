"""Write a timed video-only track, then stream-copy it alongside source streams."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any

from trtvideo.video.geometry import DEFAULT_GEOMETRY, DisplayGeometry
from trtvideo.video.metadata import EMPTY_TRACK_PROPERTIES, VideoTrackProperties
from trtvideo.video.nvcodec.encoder import NvencPacket
from trtvideo.video.output.preservation import (
    MediaPreservationError,
    build_ffmpeg_stream_copy_args,
    muxed_track_properties,
    rotation_args,
    verify_video_track,
    video_track_args,
)
from trtvideo.video.probe import probe_video


class TimedVideoTrack:
    def __init__(
        self, path: Path, codec: str, width: int, height: int, rate: Fraction, base: Fraction
    ) -> None:
        import PyNvVideoCodec as nvc

        self.path = path
        # The MOV muxer may raise a small track timescale when writing its
        # header. Pick an exact multiple up front so packet ticks stay exact.
        scale = base.denominator
        while scale < 16384:
            scale *= 2
        self._ticks = base.numerator * (scale // base.denominator)
        self._mux: Any = nvc.FFmpegMuxer(
            str(path),
            nvc.MEDIA_FORMAT.MP4,
            codec,
            width,
            height,
            rate.numerator,
            rate.denominator,
            1,
            scale,
        )

    def write(self, packet: NvencPacket, pts: int, duration: int) -> None:
        ticks = pts * self._ticks
        if not self._mux.Mux(
            packet["data"],
            ticks,
            ticks,
            duration * self._ticks,
            0,
            int(packet["picture_type"] in (2, 3)),
            packet["timestamp"],
        ):
            raise MediaPreservationError("Cannot write timestamped NVENC packet")

    def close(self) -> None:
        self._mux.Finalize()
        self._mux = None  # Close the AVFormatContext and write the MP4 trailer.


def remux_track(
    track: Path,
    source: Path,
    output: Path,
    *,
    start: Fraction,
    end: Fraction | None,
    color_args: list[str],
    geometry: DisplayGeometry = DEFAULT_GEOMETRY,
    output_size: tuple[int, int] | None = None,
    track_properties: VideoTrackProperties = EMPTY_TRACK_PROPERTIES,
) -> None:
    # The video-only track starts at zero. Apply a common origin shift only
    # for negative video starts; never reset the audio clock independently.
    shift = max(Fraction(0), -start)
    rotation_in, geometry_out = rotation_args(geometry, output)
    expected = muxed_track_properties(track_properties, output)
    if geometry != DEFAULT_GEOMETRY and output_size is None:
        raise MediaPreservationError("Output dimensions required to preserve display geometry")
    if geometry.sar != 1 and output_size is not None:
        dar = Fraction(*output_size) * geometry.sar
        geometry_out += ["-aspect:v:0", f"{dar.numerator}:{dar.denominator}"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-copyts",
        "-itsoffset",
        f"{float(start + shift):.9f}",
        *rotation_in,
        "-i",
        str(track),
        *(["-itsoffset", f"{float(shift):.9f}"] if shift else []),
        "-i",
        str(source),
        *build_ffmpeg_stream_copy_args(preserve_chapters=end is None),
        *color_args,
        *geometry_out,
        *video_track_args(expected),
        "-avoid_negative_ts",
        "disabled",
    ]
    if end is not None:
        command.extend(["-to", f"{float(end + shift):.9f}"])
    if output.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        # The default 1 kHz movie edit-list clock rounds a nonzero video start
        # to milliseconds even when the video track itself is more precise.
        scale = lcm(1000, start.denominator, end.denominator if end is not None else 1)
        if scale > 2**31 - 1:
            scale = start.denominator
        command.extend(["-movflags", "+faststart", "-movie_timescale", str(scale)])
    result = subprocess.run([*command, str(output)], capture_output=True, text=True)
    if result.returncode:
        raise MediaPreservationError(f"Cannot remux timestamped video: {result.stderr.strip()}")
    observed = probe_video(output)
    verify_video_track(expected, observed.track_properties, output)
    if geometry != DEFAULT_GEOMETRY and (
        (observed.width, observed.height) != output_size or observed.geometry != geometry
    ):
        raise MediaPreservationError("Output container did not preserve video display geometry")
