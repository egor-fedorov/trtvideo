"""Media-preservation policy and output-container preflight."""

from __future__ import annotations

import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

from trtvideo.video.geometry import DisplayGeometry


class MediaPreservationError(RuntimeError):
    """Raised when the requested output cannot satisfy the media contract."""


@cache
def has_display_rotation_option() -> bool:
    try:
        help_text = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MediaPreservationError("Cannot check FFmpeg rotation support") from exc
    return "-display_rotation" in help_text.stdout


def rotation_args(geometry: DisplayGeometry, output: Path) -> tuple[list[str], list[str]]:
    """Input and output options for a stream copy, also checked before GPU startup."""
    if geometry == DisplayGeometry():
        return [], []
    suffix = output.suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".mkv"}:
        raise MediaPreservationError(
            f"Cannot guarantee display geometry in {suffix}; use MP4 or MOV"
        )
    if not geometry.rotation:
        return [], []
    if has_display_rotation_option():
        return ["-display_rotation:v:0", str(geometry.rotation)], []
    if suffix == ".mkv":
        raise MediaPreservationError(
            "Rotated MKV requires FFmpeg with -display_rotation; use MP4 or MOV"
        )
    return [], ["-metadata:s:v:0", f"rotate={geometry.rotation}"]


def build_ffmpeg_stream_copy_args(
    *,
    video_input_index: int = 0,
    source_input_index: int = 1,
    preserve_chapters: bool = True,
) -> list[str]:
    """Build output arguments that replace video and copy all non-video streams."""
    args = ["-map", f"{video_input_index}:v:0"]
    for stream_type in ("a", "s", "d", "t"):
        args.extend(["-map", f"{source_input_index}:{stream_type}?"])
    args.extend(
        [
            "-map_metadata",
            str(source_input_index),
            "-map_chapters",
            str(source_input_index) if preserve_chapters else "-1",
            "-c",
            "copy",
        ]
    )
    return args


def build_container_preflight_command(
    *,
    input_path: str,
    output_path: str,
    preserve_chapters: bool,
) -> list[str]:
    """Build a short remux that validates copied streams against the output muxer."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:r=25:d=0.04",
        "-i",
        input_path,
        *build_ffmpeg_stream_copy_args(preserve_chapters=preserve_chapters),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        "1",
        "-t",
        "0.04",
        output_path,
    ]


def preflight_output_container(
    input_path: str,
    output_path: str,
    *,
    preserve_chapters: bool,
) -> None:
    """Fail before inference when copied streams are incompatible with the output."""
    input_resolved = Path(input_path).resolve()
    output_resolved = Path(output_path).resolve()
    if input_resolved == output_resolved:
        raise MediaPreservationError("input and output paths must be different")

    suffix = output_resolved.suffix
    if not suffix:
        raise MediaPreservationError(
            "output path must have a container extension such as .mkv or .mp4"
        )

    fd, preflight_path = tempfile.mkstemp(prefix="trtvideo-preflight-", suffix=suffix)
    os.close(fd)
    try:
        command = build_container_preflight_command(
            input_path=input_path,
            output_path=preflight_path,
            preserve_chapters=preserve_chapters,
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return

        details = result.stderr.strip() or "ffmpeg returned no error details"
        recommendation = (
            "Use an .mkv output or remove/convert the incompatible source stream."
            if suffix.lower() != ".mkv"
            else "Remove or convert the incompatible source stream."
        )
        raise MediaPreservationError(
            f"output container cannot preserve every source stream:\n{details}\n{recommendation}"
        )
    except FileNotFoundError as exc:
        raise MediaPreservationError("ffmpeg executable was not found") from exc
    finally:
        Path(preflight_path).unlink(missing_ok=True)
