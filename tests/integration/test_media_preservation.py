import json
import subprocess
from pathlib import Path

import pytest

from ai_media.video.preservation import (
    MediaPreservationError,
    ffmpeg_preservation_args,
    validate_media_preservation,
)

pytestmark = pytest.mark.docker


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _create_rich_source(tmp_path: Path) -> Path:
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:00,800\nPreserved subtitle\n",
        encoding="utf-8",
    )
    attachment = tmp_path / "attachment.txt"
    attachment.write_text("preserved attachment\n", encoding="utf-8")
    metadata = tmp_path / "metadata.ffmeta"
    metadata.write_text(
        """;FFMETADATA1
title=Preservation Fixture
comment=Global metadata
[CHAPTER]
TIMEBASE=1/1000
START=0
END=500
title=First
[CHAPTER]
TIMEBASE=1/1000
START=500
END=1000
title=Second
""",
        encoding="utf-8",
    )

    source = tmp_path / "source.mkv"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=1",
            "-i",
            str(subtitle),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:s:0",
            "-map_metadata",
            "4",
            "-map_chapters",
            "4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-c:s",
            "srt",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=jpn",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "0",
            "-disposition:s:0",
            "forced",
            "-attach",
            str(attachment),
            "-metadata:s:t:0",
            "mimetype=text/plain",
            "-metadata:s:t:0",
            f"filename={attachment.name}",
            str(source),
        ]
    )
    return source


def _probe(path: Path) -> dict:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_chapters",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def test_mkv_preserves_source_media_contract(tmp_path: Path) -> None:
    source = _create_rich_source(tmp_path)
    output = tmp_path / "output.mkv"

    validate_media_preservation(str(source), str(output), preserve_chapters=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=10:duration=1",
            "-i",
            str(source),
            *ffmpeg_preservation_args(),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )

    probe = _probe(output)
    stream_types = [stream["codec_type"] for stream in probe["streams"]]
    assert stream_types.count("video") == 1
    assert stream_types.count("audio") == 2
    assert stream_types.count("subtitle") == 1
    assert stream_types.count("attachment") == 1
    assert len(probe["chapters"]) == 2
    assert [chapter["tags"]["title"] for chapter in probe["chapters"]] == ["First", "Second"]

    format_tags = {key.lower(): value for key, value in probe["format"]["tags"].items()}
    assert format_tags["title"] == "Preservation Fixture"
    assert format_tags["comment"] == "Global metadata"

    audio_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
    audio_languages = [stream.get("tags", {}).get("language") for stream in audio_streams]
    assert audio_languages == ["eng", "jpn"]
    assert audio_streams[0]["disposition"]["default"] == 1
    assert audio_streams[1]["disposition"]["default"] == 0

    subtitle_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "subtitle"
    )
    assert subtitle_stream["disposition"]["forced"] == 1

    attachment_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "attachment"
    )
    assert attachment_stream["tags"]["filename"] == "attachment.txt"
    assert attachment_stream["tags"]["mimetype"] == "text/plain"

    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ]
    )


def test_mp4_preflight_rejects_unrepresentable_source_streams(tmp_path: Path) -> None:
    source = _create_rich_source(tmp_path)

    with pytest.raises(MediaPreservationError, match=r"Use an \.mkv output"):
        validate_media_preservation(
            str(source),
            str(tmp_path / "output.mp4"),
            preserve_chapters=True,
        )
