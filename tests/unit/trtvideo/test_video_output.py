from pathlib import Path
from types import SimpleNamespace

import pytest

import trtvideo.video.output as video_output
from trtvideo.video.output import (
    MediaPreservationError,
    build_container_preflight_command,
    build_ffmpeg_stream_copy_args,
    commit_atomic_output,
    create_staging_output,
    preflight_output_container,
)


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
