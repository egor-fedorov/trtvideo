"""Output-container preservation and atomic commit helpers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO


class MediaPreservationError(RuntimeError):
    """Raised when the requested output cannot satisfy the media contract."""


class StreamingFfmpegMuxer:
    """Own an FFmpeg mux process that consumes an elementary video stream."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        stderr_file: BinaryIO,
    ) -> None:
        if process.stdin is None:
            raise ValueError("FFmpeg mux process must expose binary stdin")
        self._process = process
        self._stderr_file = stderr_file
        self._input_closed = False
        self._finished = False

    @classmethod
    def start(cls, command: Sequence[str]) -> StreamingFfmpegMuxer:
        """Start FFmpeg without creating a stderr pipe that could deadlock."""
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                bufsize=0,
            )
        except OSError as exc:
            stderr_file.close()
            raise MediaPreservationError(f"cannot start ffmpeg mux process: {exc}") from exc
        return cls(process, stderr_file)

    def _read_stderr(self) -> str:
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        return self._stderr_file.read().decode("utf-8", errors="replace").strip()

    def _complete_process(self) -> tuple[int, str]:
        returncode = self._process.wait()
        details = self._read_stderr()
        self._stderr_file.close()
        self._finished = True
        return returncode, details

    def write(self, data: bytes | bytearray | memoryview) -> None:
        """Write a complete encoded packet, handling partial pipe writes."""
        if not data:
            return
        if self._finished or self._input_closed:
            raise MediaPreservationError("cannot write to a finished ffmpeg mux process")

        stdin = self._process.stdin
        if stdin is None:
            raise MediaPreservationError("ffmpeg mux stdin is unavailable")

        remaining = memoryview(data)
        try:
            while remaining:
                written = stdin.write(remaining)
                if written is None or written <= 0:
                    raise BrokenPipeError("ffmpeg mux pipe accepted no data")
                remaining = remaining[written:]
        except (BrokenPipeError, OSError) as exc:
            returncode, details = self._complete_process()
            message = details or "ffmpeg returned no error details"
            raise MediaPreservationError(
                f"ffmpeg mux pipe failed with exit code {returncode}:\n{message}"
            ) from exc

    def close_input(self) -> None:
        """Signal end-of-stream without waiting for container finalization."""
        if self._finished or self._input_closed:
            return
        stdin = self._process.stdin
        if stdin is None:
            raise MediaPreservationError("ffmpeg mux stdin is unavailable")
        try:
            stdin.close()
        except (BrokenPipeError, OSError) as exc:
            returncode, details = self._complete_process()
            message = details or "ffmpeg returned no error details"
            raise MediaPreservationError(
                f"ffmpeg mux input close failed with exit code {returncode}:\n{message}"
            ) from exc
        self._input_closed = True

    def finish(self) -> None:
        """Close the input stream and require successful FFmpeg completion."""
        if self._finished:
            return
        self.close_input()
        returncode, details = self._complete_process()
        if returncode != 0:
            message = details or "ffmpeg returned no error details"
            raise MediaPreservationError(
                f"ffmpeg mux failed with exit code {returncode}:\n{message}"
            )

    def abort(self) -> None:
        """Stop an unfinished mux process during pipeline cleanup."""
        if self._finished:
            return
        stdin = self._process.stdin
        if stdin is not None and not stdin.closed:
            with suppress(OSError):
                stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._read_stderr()
        self._stderr_file.close()
        self._finished = True


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


def build_ffmpeg_streaming_mux_command(
    *,
    video_codec: str,
    fps: str,
    source_input_path: str,
    output_path: str,
    preserve_chapters: bool,
    color_metadata_args: Sequence[str] = (),
    duration_args: Sequence[str] = (),
    faststart: bool = False,
) -> list[str]:
    """Build an FFmpeg command that muxes encoded video received through stdin."""
    if video_codec not in {"h264", "hevc"}:
        raise ValueError(f"Unsupported elementary video codec: {video_codec}")

    faststart_args = ["-movflags", "+faststart"] if faststart else []
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        video_codec,
        "-r",
        fps,
        "-i",
        "pipe:0",
        "-i",
        source_input_path,
        *build_ffmpeg_stream_copy_args(preserve_chapters=preserve_chapters),
        *color_metadata_args,
        *duration_args,
        *faststart_args,
        output_path,
    ]


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
