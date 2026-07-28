"""Video metadata contract and ffprobe adapter."""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from trtvideo.video.fps import parse_fps


@dataclass(frozen=True)
class VideoInfo:
    """Video stream metadata required by the production pipeline."""

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


def _parse_optional_int(value: Any) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def probe_video(input_path: str) -> VideoInfo:
    """Read the primary video stream metadata through ffprobe.

    Args:
        input_path: Path to the video file.

    Returns:
        VideoInfo with dimensions, frame rate, frame count and color metadata.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed: {result.stderr}")
        sys.exit(1)

    info = json.loads(result.stdout)
    container_bit_rate = _parse_optional_int(info.get("format", {}).get("bit_rate"))

    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            fps_str = stream.get("r_frame_rate", "30/1")
            fps = parse_fps(fps_str)

            nb_frames = int(stream.get("nb_frames") or 0)
            width = int(stream["width"])
            height = int(stream["height"])

            return VideoInfo(
                fps=fps,
                fps_str=fps_str,
                nb_frames=nb_frames,
                width=width,
                height=height,
                pix_fmt=stream.get("pix_fmt"),
                color_range=stream.get("color_range"),
                color_space=stream.get("color_space"),
                color_transfer=stream.get("color_transfer"),
                color_primaries=stream.get("color_primaries"),
                video_bit_rate=_parse_optional_int(stream.get("bit_rate")),
                container_bit_rate=container_bit_rate,
            )

    print("ERROR: No video stream found")
    sys.exit(1)
