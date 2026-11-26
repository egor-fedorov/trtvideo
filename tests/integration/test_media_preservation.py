import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from trtvideo.compatibility.input import (
    InputPreparation,
    input_manifest_evidence,
    prepare_input,
)
from trtvideo.demo import media as demo_media
from trtvideo.demo.config import (
    DEMO_AUDIO_CHANNELS,
    DEMO_AUDIO_SAMPLE_RATE_HZ,
    DEMO_FRAMES,
    DEMO_INPUT_HEIGHT,
    DEMO_INPUT_WIDTH,
    VIDEO_DURATION_SECONDS,
    DemoPaths,
    DemoVideoContract,
)
from trtvideo.demo.media import (
    build_demo_input_command,
    validate_demo_video,
)
from trtvideo.video.color import SdrColorContract
from trtvideo.video.geometry import DisplayGeometry
from trtvideo.video.output import (
    MediaPreservationError,
    build_ffmpeg_stream_copy_args,
    preflight_output_container,
)
from trtvideo.video.output.preservation import has_display_rotation_option
from trtvideo.video.output.timed_track import remux_track
from trtvideo.video.probe import probe_video

pytestmark = pytest.mark.docker

_MIN_BROADBAND_AUDIO_SI_SDR_DB = 20.0


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _decode_stereo_pcm(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-t",
            str(VIDEO_DURATION_SECONDS),
            "-vn",
            "-ac",
            str(DEMO_AUDIO_CHANNELS),
            "-ar",
            str(DEMO_AUDIO_SAMPLE_RATE_HZ),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    assert samples.size % DEMO_AUDIO_CHANNELS == 0
    return samples.reshape((-1, DEMO_AUDIO_CHANNELS))


def _minimum_audio_si_sdr_db(reference_path: Path, candidate_path: Path) -> float:
    reference = _decode_stereo_pcm(reference_path)
    candidate = _decode_stereo_pcm(candidate_path)
    compared_frames = min(len(reference), len(candidate))
    assert compared_frames > 0
    reference = reference[:compared_frames].astype(np.float64)
    candidate = candidate[:compared_frames].astype(np.float64)
    reference_energy = np.sum(reference**2, axis=0)
    assert np.all(reference_energy > 0)
    scale = np.sum(reference * candidate, axis=0) / reference_energy
    target = reference * scale
    distortion = candidate - target
    scores = 10 * np.log10(np.sum(target**2, axis=0) / np.sum(distortion**2, axis=0))
    return float(np.min(scores))


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


def test_timed_remux_keeps_all_vfr_pts_longer_audio_and_faststart(tmp_path: Path) -> None:
    source, track, out = (tmp_path / name for name in ("source.mp4", "track.mp4", "out.mp4"))
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=128x72:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=44100:duration=4",
            "-vf",
            r"select=lt(n\,48)+not(mod(n\,2))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    _run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0", "-c", "copy", str(track)])
    remux_track(track, source, out, start=Fraction(0), end=None, color_args=[])

    for stream, duration in (("v:0", Fraction(95, 24)), ("a:0", Fraction(4))):

        def packets(path, stream=stream, duration=duration):
            result = _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    stream,
                    "-show_packets",
                    "-show_streams",
                    "-show_data_hash",
                    "sha256",
                    "-show_entries",
                    "stream=time_base,duration_ts:packet=pts,duration,data_hash",
                    "-of",
                    "json",
                    str(path),
                ]
            )
            data = json.loads(result.stdout)
            info = data["streams"][0]
            base = Fraction(info["time_base"])
            assert base * int(info["duration_ts"]) == duration
            return [
                (base * int(p["pts"]), base * int(p["duration"]), p["data_hash"])
                for p in data["packets"]
            ]

        original = packets(source)
        assert packets(out) == original
        if stream == "v:0":
            assert len(original) == 72
    atoms = out.read_bytes()
    assert atoms.index(b"moov") < atoms.index(b"mdat")
    _run(["ffmpeg", "-v", "error", "-xerror", "-i", str(out), "-f", "null", "-"])


def test_self_contained_demo_input_passes_its_media_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = DemoPaths.under(tmp_path)
    paths.source_video.parent.mkdir(parents=True)
    paths.input_video.parent.mkdir(parents=True)
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
            "testsrc2=size=320x180:rate=24:duration=5",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=48000:duration=5:seed=12345",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-ac",
            "2",
            "-f",
            "nut",
            str(paths.source_video),
        ]
    )
    monkeypatch.setattr(demo_media, "VIDEO_START_SECONDS", 0)

    _run(build_demo_input_command(paths))
    observed = validate_demo_video(
        paths.input_video,
        DemoVideoContract(DEMO_INPUT_WIDTH, DEMO_INPUT_HEIGHT),
    )
    minimum_audio_si_sdr_db = _minimum_audio_si_sdr_db(
        paths.source_video,
        paths.input_video,
    )

    assert observed["width"] == 1280
    assert observed["height"] == 720
    assert observed["frames"] == DEMO_FRAMES
    assert minimum_audio_si_sdr_db >= _MIN_BROADBAND_AUDIO_SI_SDR_DB


def test_compatibility_input_is_decodable_and_bound_to_its_manifest(tmp_path: Path) -> None:
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
            "testsrc2=size=128x72:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    rotated = tmp_path / "custom.mp4"
    modern = has_display_rotation_option()
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            *(["-display_rotation", "90"] if modern else []),
            "-i",
            str(source),
            "-c",
            "copy",
            "-bsf:v",
            "h264_metadata=sample_aspect_ratio=4/3",
            "-aspect",
            "64:27",
            *([] if modern else ["-metadata:s:v:0", "rotate=90"]),
            str(rotated),
        ]
    )
    assert probe_video(rotated).geometry == DisplayGeometry(90, Fraction(4, 3))
    request = InputPreparation(
        source=rotated,
        output=tmp_path / "input.mp4",
        manifest=tmp_path / "input.json",
        width=128,
        height=72,
        frames=24,
    )

    manifest = prepare_input(request)
    evidence = input_manifest_evidence(request.manifest, request.output)

    assert manifest["observed"]["frames"] == 24
    assert manifest["observed"]["full_decode"] is True
    assert evidence["prepared_input"] == manifest["output"]
    assert probe_video(request.output).geometry == probe_video(rotated).geometry


@pytest.mark.parametrize("codec,encoder", [("h264", "libx264"), ("hevc", "libx265")])
@pytest.mark.parametrize("ext", ["mp4", "mov", "mkv"])
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_timed_remux_display_geometry(tmp_path, codec, encoder, ext, angle):
    track, out = tmp_path / "track.mp4", tmp_path / f"out.{ext}"
    geometry = DisplayGeometry(angle, Fraction(4, 3))
    if angle and ext == "mkv" and not has_display_rotation_option():
        with pytest.raises(MediaPreservationError, match="use MP4 or MOV"):
            remux_track(
                track,
                track,
                out,
                start=Fraction(0),
                end=None,
                color_args=[],
                geometry=geometry,
                output_size=(128, 72),
            )
        return
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=128x72:r=24:d=0.15",
            "-c:v",
            encoder,
            "-threads",
            "1",
            *(["-x265-params", "pools=none:frame-threads=1"] if codec == "hevc" else []),
            str(track),
        ]
    )
    color = SdrColorContract("tv", "bt709", "bt709", "bt709")
    remux_track(
        track,
        track,
        out,
        start=Fraction(0),
        end=None,
        color_args=color.ffmpeg_args() + color.bitstream_args(codec, sar=geometry.sar),
        geometry=geometry,
        output_size=(128, 72),
    )
    assert probe_video(out).geometry == geometry

    def pixel_hash(path):
        return _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-noautorotate",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "hash",
                "-",
            ]
        ).stdout

    assert pixel_hash(track) == pixel_hash(out)
