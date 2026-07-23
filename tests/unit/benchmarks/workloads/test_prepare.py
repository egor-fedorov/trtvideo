from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.scripts.workloads.prepare import (
    WorkloadError,
    build_ffmpeg_command,
    build_model_commands,
    download_source,
    load_manifest,
    sha256_file,
    validate_manifest,
    validate_video_probe,
    verify_source_file,
)

MANIFEST_PATH = Path("benchmarks/workloads/realesrgan_x2plus_sintel.json")
SPAN_MANIFEST_PATH = Path("benchmarks/workloads/liveaction_span_sintel.json")


@pytest.fixture
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_is_valid(manifest: dict) -> None:
    validate_manifest(manifest)


def test_span_manifest_is_valid() -> None:
    validate_manifest(json.loads(SPAN_MANIFEST_PATH.read_text(encoding="utf-8")))


def test_manifest_rejects_non_sha256(manifest: dict) -> None:
    invalid = copy.deepcopy(manifest)
    invalid["clip"]["source"]["sha256"] = "pending"

    with pytest.raises(WorkloadError, match="lowercase SHA256"):
        validate_manifest(invalid)


def test_manifest_rejects_path_outside_repository(manifest: dict) -> None:
    invalid = copy.deepcopy(manifest)
    invalid["model"]["weights_path"] = "../weights.pth"

    with pytest.raises(WorkloadError, match="inside the repository"):
        validate_manifest(invalid)


def test_manifest_rejects_missing_measurement_contract(manifest: dict) -> None:
    invalid = copy.deepcopy(manifest)
    del invalid["benchmark"]["warmup_frames"]

    with pytest.raises(WorkloadError, match="benchmark fields are missing"):
        validate_manifest(invalid)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(WorkloadError, match="Cannot read workload manifest"):
        load_manifest(path)


def test_sha256_file_streams_file(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"benchmark")

    assert sha256_file(path) == "0e89820860c342f2c7ec694d144023b10301c2accdd078cb5167a06d0c3d5bcc"


def test_verify_source_file_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"benchmark")
    source = {"size_bytes": 9, "sha256": "0" * 64}

    with pytest.raises(WorkloadError, match="Checksum mismatch"):
        verify_source_file(path, source)


def test_download_source_reuses_valid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"benchmark")
    source = {
        "url": "https://example.invalid/asset.bin",
        "size_bytes": 9,
        "sha256": "0e89820860c342f2c7ec694d144023b10301c2accdd078cb5167a06d0c3d5bcc",
    }

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("valid existing source must not trigger a download")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    download_source(source, path, force=False)


def test_download_source_promotes_complete_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.part"
    partial.write_bytes(b"benchmark")
    source = {
        "url": "https://example.invalid/asset.bin",
        "size_bytes": 9,
        "sha256": "0e89820860c342f2c7ec694d144023b10301c2accdd078cb5167a06d0c3d5bcc",
    }

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("complete partial source must not trigger a download")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    download_source(source, path, force=False)

    assert path.read_bytes() == b"benchmark"
    assert not partial.exists()


def test_build_ffmpeg_command_pins_media_contract(manifest: dict, tmp_path: Path) -> None:
    variant = manifest["clip"]["variants"][0]
    command = build_ffmpeg_command(
        manifest,
        tmp_path / "source.y4m",
        variant,
        tmp_path / "output.mp4",
    )

    assert command[0] == "ffmpeg"
    assert "scale=1280:720:flags=lanczos,setsar=1" in command
    assert command[command.index("-frames:v") + 1] == "1000"
    assert command[command.index("-r") + 1] == "24/1"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-x264-params") + 1] == (
        "keyint=24:min-keyint=24:scenecut=0:bframes=0:"
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"
    )


def test_build_model_commands_use_static_variants(manifest: dict, tmp_path: Path) -> None:
    commands = build_model_commands(manifest, tmp_path)

    assert commands[0][0] == "export-onnx"
    assert commands[0][commands[0].index("--name") + 1] == "realesrgan_x2plus"
    assert [command[0] for command in commands[1:]] == ["prepare-onnx", "prepare-onnx"]
    assert all("fp16" in command for command in commands[1:])


def test_validate_video_probe_accepts_canonical_clip(manifest: dict) -> None:
    clip = manifest["clip"]
    variant = clip["variants"][0]
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "nb_frames": "1000",
                "nb_read_frames": "1000",
                "has_b_frames": 0,
                "sample_aspect_ratio": "1:1",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            }
        ],
        "format": {"duration": "41.666667"},
    }

    validate_video_probe(probe, variant=variant, clip=clip)


def test_validate_video_probe_rejects_old_b_frame_clip(manifest: dict) -> None:
    clip = manifest["clip"]
    variant = clip["variants"][0]
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "nb_frames": "1000",
                "nb_read_frames": "1000",
                "has_b_frames": 2,
                "sample_aspect_ratio": "1:1",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            }
        ],
        "format": {"duration": "41.666667"},
    }

    with pytest.raises(WorkloadError, match="has_b_frames"):
        validate_video_probe(probe, variant=variant, clip=clip)


def test_validate_video_probe_rejects_extra_stream(manifest: dict) -> None:
    clip = manifest["clip"]
    variant = clip["variants"][0]
    probe = {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]}

    with pytest.raises(WorkloadError, match="exactly one stream"):
        validate_video_probe(probe, variant=variant, clip=clip)
