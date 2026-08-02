"""FFprobe adapter for normalized video metadata."""

from __future__ import annotations

import json
import subprocess
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
            nb_frames=int(stream.get("nb_frames") or 0),
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
