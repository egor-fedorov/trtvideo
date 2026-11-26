"""FFprobe adapter for normalized video metadata."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from trtvideo.video.fps import parse_fps
from trtvideo.video.metadata import VideoMetadata


class VideoProbeError(RuntimeError):
    """Raised when FFprobe cannot produce valid primary-stream metadata."""


def _parse_optional_int(value: Any) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _video_metadata(stream: dict[str, Any], container_bit_rate: int | None) -> VideoMetadata:
    try:
        fps_str = str(stream.get("r_frame_rate", "30/1"))
        return VideoMetadata(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=parse_fps(fps_str),
            fps_str=fps_str,
            nb_frames=_parse_optional_int(stream.get("nb_frames")) or 0,
            time_base=Fraction(stream["time_base"]) if stream.get("time_base") else None,
            start_pts=int(stream["start_pts"])
            if stream.get("start_pts") not in (None, "N/A")
            else None,
            duration_ts=_parse_optional_int(stream.get("duration_ts")),
            pix_fmt=stream.get("pix_fmt"),
            color_range=stream.get("color_range"),
            color_space=stream.get("color_space"),
            color_transfer=stream.get("color_transfer"),
            color_primaries=stream.get("color_primaries"),
            video_bit_rate=_parse_optional_int(stream.get("bit_rate")),
            container_bit_rate=container_bit_rate,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise VideoProbeError(f"FFprobe returned invalid video metadata: {exc}") from exc


def probe_video(input_path: str | Path) -> VideoMetadata:
    """Read and normalize the primary video stream metadata through FFprobe."""
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise VideoProbeError(f"Cannot run ffprobe: {exc}") from exc
    if result.returncode != 0:
        details = result.stderr.strip() or "ffprobe returned no error details"
        raise VideoProbeError(f"FFprobe failed: {details}")

    try:
        payload: Any = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"FFprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VideoProbeError("FFprobe output must be a JSON object")

    format_payload = payload.get("format", {})
    if not isinstance(format_payload, dict):
        raise VideoProbeError("FFprobe format metadata must be a JSON object")
    container_bit_rate = _parse_optional_int(format_payload.get("bit_rate"))

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoProbeError("FFprobe output does not contain a stream list")
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return _video_metadata(stream, container_bit_rate)
    raise VideoProbeError("No video stream found")


def _packet_times(path: Path, interval: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            interval,
            "-show_packets",
            "-show_entries",
            "packet=pts,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise VideoProbeError(f"Cannot probe video packet timestamps: {result.stderr.strip()}")
    return json.loads(result.stdout).get("packets", [])


def validate_video_clock(path: Path, info: VideoMetadata) -> None:
    if info.time_base is None or info.time_base <= 0:
        raise VideoProbeError("Input video has no valid presentation time base")
    if info.start_pts is None and not any(
        isinstance(packet.get("pts"), int) for packet in _packet_times(path, "%+#16")
    ):
        raise VideoProbeError(
            "Input video has no presentation timestamps; use a timestamped container"
        )


def last_frame_end(path: Path, info: VideoMetadata, pts: int) -> int:
    """Get the end of the video stream, not the (possibly longer) audio track."""
    if info.start_pts is not None and info.duration_ts is not None:
        end = info.start_pts + info.duration_ts
        if end > pts:
            return end
    # Matroska commonly has no stream duration. Seek to the last displayed
    # frame's GOP and match its packet by PTS, not by decode order.
    assert info.time_base is not None
    for packet in _packet_times(path, f"{float(pts * info.time_base):.6f}%"):
        if str(packet.get("pts")) == str(pts):
            duration = _parse_optional_int(packet.get("duration"))
            if duration is not None:
                return pts + duration
    raise VideoProbeError(f"Cannot determine the duration of the last video frame at PTS {pts}")
