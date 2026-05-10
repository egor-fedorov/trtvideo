"""Video metadata retrieval via ffprobe."""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoInfo:
    """Minimal video metadata contract used by the current pipelines."""

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

    def __getitem__(self, key: str) -> Any:
        """Preserve existing dict-style call sites during the contracts split."""
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


def _parse_fps(value: str) -> float:
    num, den = map(int, value.split("/"))
    return num / den if den else 0.0


def get_video_info(input_path: str) -> VideoInfo:
    """Get video metadata via ffprobe.

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
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed: {result.stderr}")
        sys.exit(1)

    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            fps_str = stream.get("r_frame_rate", "30/1")
            fps = _parse_fps(fps_str)

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
            )

    print("ERROR: No video stream found")
    sys.exit(1)
