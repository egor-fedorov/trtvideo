from pathlib import Path
from types import SimpleNamespace

import pytest

import trtvideo.video.output as video_output
from trtvideo.video.output import (
    MediaPreservationError,
    StreamingFfmpegMuxer,
    build_container_preflight_command,
    build_ffmpeg_stream_copy_args,
    build_ffmpeg_streaming_mux_command,
    commit_atomic_output,
    create_staging_output,
    preflight_output_container,
)


class _FakeStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes | bytearray | memoryview) -> int:
        chunk = bytes(data[:3])
        self.data.extend(chunk)
        return len(chunk)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.returncode = returncode
        self.waited = False
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode if self.waited else None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_stream_copy_args_include_every_non_video_stream() -> None:
    assert build_ffmpeg_stream_copy_args() == [
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map",
        "1:d?",
        "-map",
        "1:t?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c",
        "copy",
    ]


def test_stream_copy_args_omit_chapters_for_shortened_output() -> None:
    args = build_ffmpeg_stream_copy_args(preserve_chapters=False)

    assert args[args.index("-map_chapters") + 1] == "-1"


def test_preflight_uses_selected_container_and_preservation_contract() -> None:
    command = build_container_preflight_command(
        input_path="input.mkv",
        output_path="probe.mp4",
        preserve_chapters=True,
    )

    assert command[-1] == "probe.mp4"
    assert command[command.index("1:a?") - 1 : command.index("1:a?") + 1] == ["-map", "1:a?"]
    assert command[command.index("1:t?") - 1 : command.index("1:t?") + 1] == ["-map", "1:t?"]
    assert command[command.index("-map_chapters") + 1] == "1"


def test_streaming_mux_command_preserves_output_contract() -> None:
    command = build_ffmpeg_streaming_mux_command(
        video_codec="h264",
        fps="24000/1001",
        source_input_path="input.mkv",
        output_path="output.mp4",
        preserve_chapters=True,
        color_metadata_args=("-color_range", "tv"),
        duration_args=("-t", "10.000000"),
        faststart=True,
    )

    assert command[command.index("-f") + 1] == "h264"
    assert command[command.index("-r") + 1] == "24000/1001"
    assert command[command.index("pipe:0") + 1 : command.index("pipe:0") + 3] == [
        "-i",
        "input.mkv",
    ]
    assert command[command.index("-map_chapters") + 1] == "1"
    assert command[command.index("-color_range") + 1] == "tv"
    assert command[command.index("-t") + 1] == "10.000000"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command[-1] == "output.mp4"


def test_streaming_muxer_handles_partial_pipe_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(
        video_output.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    muxer = StreamingFfmpegMuxer.start(["ffmpeg"])
    muxer.write(b"encoded-packet")
    muxer.close_input()
    muxer.finish()

    assert process.stdin.data == b"encoded-packet"
    assert process.stdin.closed
    assert process.waited


def test_streaming_muxer_reports_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=2)

    def fake_popen(*_args, **kwargs):
        kwargs["stderr"].write(b"invalid output container")
        return process

    monkeypatch.setattr(video_output.subprocess, "Popen", fake_popen)

    muxer = StreamingFfmpegMuxer.start(["ffmpeg"])
    with pytest.raises(MediaPreservationError, match="invalid output container"):
        muxer.finish()


def test_streaming_muxer_terminates_unfinished_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(
        video_output.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    muxer = StreamingFfmpegMuxer.start(["ffmpeg"])
    muxer.abort()

    assert process.stdin.closed
    assert process.terminated
    assert process.waited
    assert not process.killed


def test_preflight_recommends_mkv_for_incompatible_mp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_output.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stderr="codec not supported in container",
        ),
    )

    with pytest.raises(MediaPreservationError, match=r"Use an \.mkv output"):
        preflight_output_container(
            "input.mkv",
            "output.mp4",
            preserve_chapters=True,
        )


def test_preflight_rejects_overwriting_input() -> None:
    with pytest.raises(MediaPreservationError, match="must be different"):
        preflight_output_container(
            "input.mkv",
            "./input.mkv",
            preserve_chapters=True,
        )


def test_atomic_output_replaces_target_only_after_commit(tmp_path: Path) -> None:
    output = tmp_path / "output.mkv"
    output.write_text("old", encoding="utf-8")

    temporary = create_staging_output(str(output))
    assert output.read_text(encoding="utf-8") == "old"
    temporary.write_text("complete", encoding="utf-8")

    commit_atomic_output(temporary, str(output))

    assert output.read_text(encoding="utf-8") == "complete"
    assert not temporary.exists()
