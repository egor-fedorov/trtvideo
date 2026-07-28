"""Output-container preservation and atomic commit helpers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class MediaPreservationError(RuntimeError):
    """Raised when the requested output cannot satisfy the media contract."""


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
            f"output container cannot preserve every source stream:\n{details}\n"
            f"{recommendation}"
        )
    except FileNotFoundError as exc:
        raise MediaPreservationError("ffmpeg executable was not found") from exc
    finally:
        Path(preflight_path).unlink(missing_ok=True)


def create_staging_output(output_path: str) -> Path:
    """Reserve a same-directory temporary path suitable for atomic replacement."""
    output = Path(output_path)
    parent = output.parent
    if not parent.is_dir():
        raise MediaPreservationError(f"output directory does not exist: {parent}")
    if not output.suffix:
        raise MediaPreservationError(
            "output path must have a container extension such as .mkv or .mp4"
        )

    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=f".partial{output.suffix}",
        dir=parent,
    )
    os.close(fd)
    return Path(temporary)


def commit_atomic_output(temporary_path: Path, output_path: str) -> None:
    """Atomically expose a completed output without leaving partial media behind."""
    os.replace(temporary_path, output_path)
