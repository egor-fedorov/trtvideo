"""Streaming FFmpeg mux process for encoded video packets."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from typing import BinaryIO

from trtvideo.video.output.preservation import (
    MediaPreservationError,
    build_ffmpeg_stream_copy_args,
)


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
