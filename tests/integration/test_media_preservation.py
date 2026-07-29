import json
import subprocess
from pathlib import Path

import pytest

from trtvideo.demo.config import (
    DEMO_INPUT_HEIGHT,
    DEMO_INPUT_WIDTH,
    DemoPaths,
    DemoVideoContract,
)
from trtvideo.demo.media import (
    build_demo_input_command,
    validate_demo_video,
    write_demo_media_assets,
)
from trtvideo.video.output import (
    MediaPreservationError,
    StreamingFfmpegMuxer,
    build_ffmpeg_stream_copy_args,
    build_ffmpeg_streaming_mux_command,
    preflight_output_container,
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

    preflight_output_container(str(source), str(output), preserve_chapters=True)
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
            *build_ffmpeg_stream_copy_args(),
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
        preflight_output_container(
            str(source),
            str(tmp_path / "output.mp4"),
            preserve_chapters=True,
        )


def test_streaming_mux_preserves_audio_and_faststart(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    elementary_stream = tmp_path / "video.h264"
    output = tmp_path / "output.mp4"
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
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-bsf:v",
            "h264_mp4toannexb",
            "-f",
            "h264",
            str(elementary_stream),
        ]
    )

    command = build_ffmpeg_streaming_mux_command(
        video_codec="h264",
        fps="10/1",
        source_input_path=str(source),
        output_path=str(output),
        preserve_chapters=True,
        color_metadata_args=(
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
        ),
        faststart=True,
    )
    muxer = StreamingFfmpegMuxer.start(command)
    encoded = elementary_stream.read_bytes()
    for offset in range(0, len(encoded), 4096):
        muxer.write(encoded[offset : offset + 4096])
    muxer.finish()

    probe = _probe(output)
    stream_types = [stream["codec_type"] for stream in probe["streams"]]
    assert stream_types.count("video") == 1
    assert stream_types.count("audio") == 1
    atoms = output.read_bytes()
    assert atoms.index(b"moov") < atoms.index(b"mdat")
    _run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"])


def test_self_contained_demo_input_passes_its_media_contract(tmp_path: Path) -> None:
    paths = DemoPaths.under(tmp_path)
    write_demo_media_assets(paths)
    paths.input_video.parent.mkdir(parents=True)

    _run(build_demo_input_command(paths))
    observed = validate_demo_video(
        paths.input_video,
        DemoVideoContract(DEMO_INPUT_WIDTH, DEMO_INPUT_HEIGHT),
    )

    assert observed["width"] == 1280
    assert observed["height"] == 720
    assert observed["frames"] == 24
