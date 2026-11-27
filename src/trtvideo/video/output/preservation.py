"""Media-preservation policy and output-container preflight."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path

from trtvideo.video.geometry import DisplayGeometry
from trtvideo.video.metadata import EMPTY_TRACK_PROPERTIES, VideoTrackProperties
from trtvideo.video.probe import probe_video


class MediaPreservationError(RuntimeError):
    """Raised when the requested output cannot satisfy the media contract."""


_MOV_FAMILY = {".mp4", ".m4v", ".mov"}


def muxed_track_properties(source: VideoTrackProperties, output: Path) -> VideoTrackProperties:
    if output.suffix.lower() not in _MOV_FAMILY:
        return source
    if source.title and source.handler_name and source.title != source.handler_name:
        raise MediaPreservationError(
            f"{output.suffix} cannot preserve both video title={source.title!r} "
            f"and handler_name={source.handler_name!r}. Use an .mkv output."
        )
    return replace(source, title=None, handler_name=source.handler_name or source.title)


def video_track_args(expected: VideoTrackProperties) -> list[str]:
    args = []
    for name in ("language", "title", "handler_name"):
        args.extend(["-metadata:s:v:0", f"{name}={getattr(expected, name) or ''}"])
    return [*args, "-disposition:v:0", "+".join(sorted(expected.dispositions)) or "0"]


def verify_video_track(
    expected: VideoTrackProperties, actual: VideoTrackProperties, output: Path
) -> None:
    differences = []
    for name in ("language", "title", "handler_name"):
        wanted, got = getattr(expected, name), getattr(actual, name)
        if wanted != got:
            differences.append(f"{name}={wanted!r} (got {got!r})")
    for flag in sorted(expected.dispositions ^ actual.dispositions):
        wanted_flag = flag in expected.dispositions
        # Only one video track is mapped. MOV-family muxers may enable it by default.
        if flag == "default" and not wanted_flag and output.suffix.lower() in _MOV_FAMILY:
            continue
        differences.append(f"{flag}={int(wanted_flag)} (got {int(not wanted_flag)})")
    if differences:
        advice = (
            "Use an .mkv output."
            if output.suffix.lower() != ".mkv"
            else "Choose a container that supports these properties."
        )
        raise MediaPreservationError(
            f"{output.suffix} cannot preserve video track properties: "
            f"{', '.join(differences)}. {advice}"
        )


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
    track_properties: VideoTrackProperties = EMPTY_TRACK_PROPERTIES,
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
        *video_track_args(track_properties),
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
    track_properties: VideoTrackProperties = EMPTY_TRACK_PROPERTIES,
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

    expected = muxed_track_properties(track_properties, output_resolved)
    fd, preflight_path = tempfile.mkstemp(prefix="trtvideo-preflight-", suffix=suffix)
    os.close(fd)
    try:
        command = build_container_preflight_command(
            input_path=input_path,
            output_path=preflight_path,
            preserve_chapters=preserve_chapters,
            track_properties=expected,
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            verify_video_track(
                expected, probe_video(preflight_path).track_properties, output_resolved
            )
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
