from __future__ import annotations

import json
from pathlib import Path

import pytest

from trtvideo.cli.prepare_compatibility_input import build_parser
from trtvideo.compatibility import input as compatibility_input
from trtvideo.compatibility.evidence import CompatibilityEvidenceError, file_identity
from trtvideo.compatibility.input import (
    FIXTURE_CONTRACT_VERSION,
    VIDEO_AUTHOR,
    VIDEO_LICENSE,
    VIDEO_LICENSE_URL,
    VIDEO_NAME,
    VIDEO_SHA256,
    VIDEO_SIZE_BYTES,
    VIDEO_SOURCE_PAGE_URL,
    CompatibilityInputError,
    InputPreparation,
    input_command,
    input_manifest_evidence,
    probe_video_size,
)


def test_default_input_command_preserves_live_action_contract(tmp_path: Path) -> None:
    request = InputPreparation(
        output=tmp_path / "input.mp4",
        manifest=tmp_path / "input.json",
        source_cache=tmp_path / "source.webm",
        width=640,
        height=360,
    )

    command = input_command(request)

    assert command[command.index("-ss") + 1] == "14"
    assert command[command.index("-frames:v") + 1] == "120"
    assert command[command.index("-map") + 3] == "0:a:0"
    video_filter = command[command.index("-vf") + 1]
    assert "scale=640:360:flags=lanczos" in video_filter
    assert "out_color_matrix=bt709:out_range=tv" in video_filter
    assert "bframes=0" in command[command.index("-x264-params") + 1]
    assert any(value.startswith("copyright=CC-BY-SA-4.0") for value in command)


def test_custom_input_is_normalized_without_false_attribution(tmp_path: Path) -> None:
    request = InputPreparation(
        output=tmp_path / "input.mp4",
        manifest=tmp_path / "input.json",
        source=tmp_path / "custom.mkv",
        width=1280,
        height=720,
    )

    command = input_command(request)

    assert command[command.index("-ss") + 1] == "0"
    assert command[command.index("-map") + 3] == "0:a:0?"
    assert not any(value.startswith("copyright=") for value in command)


def test_prepare_input_parser_defaults_to_120_frame_720p_fixture(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--output",
            str(tmp_path / "input.mp4"),
            "--source-cache",
            str(tmp_path / "source.webm"),
        ]
    )

    assert args.size == (1280, 720)
    assert args.frames == 120
    assert args.input is None


def test_custom_input_probe_accepts_declared_sdr_bt709(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "custom.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        compatibility_input,
        "_run_json",
        lambda _command: {
            "streams": [
                {
                    "width": 1920,
                    "height": 1080,
                    "color_space": "bt709",
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                }
            ]
        },
    )

    assert probe_video_size(source) == (1920, 1080)


def test_custom_input_probe_rejects_hdr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "custom.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        compatibility_input,
        "_run_json",
        lambda _command: {
            "streams": [
                {
                    "width": 3840,
                    "height": 2160,
                    "color_space": "bt2020nc",
                    "color_transfer": "smpte2084",
                    "color_primaries": "bt2020",
                }
            ]
        },
    )

    with pytest.raises(CompatibilityInputError, match="HDR transfer"):
        probe_video_size(source)


def test_input_manifest_evidence_binds_prepared_video_without_paths(tmp_path: Path) -> None:
    source = tmp_path / "custom.mov"
    prepared = tmp_path / "prepared.mp4"
    source.write_bytes(b"source")
    prepared.write_bytes(b"prepared")
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "document_type": "trtvideo-compatibility-input",
                "schema_version": 1,
                "fixture_contract": FIXTURE_CONTRACT_VERSION,
                "source_kind": "user-supplied",
                "source": file_identity(source),
                "output": file_identity(prepared),
                "command": ["ffmpeg", "-i", str(source), str(prepared)],
                "observed": {
                    "width": 1280,
                    "height": 720,
                    "frames": 120,
                    "fps": "24/1",
                    "audio_streams": 1,
                    "duration_sec": 5.0,
                    "timestamps": "strictly_monotonic",
                    "full_decode": True,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = input_manifest_evidence(manifest, prepared)

    assert evidence["source_kind"] == "user-supplied"
    assert evidence["prepared_input"] == file_identity(prepared)
    assert evidence["attribution"] is None
    assert str(tmp_path) not in json.dumps(evidence)


def test_input_manifest_evidence_rejects_changed_prepared_video(tmp_path: Path) -> None:
    source = tmp_path / "custom.mov"
    prepared = tmp_path / "prepared.mp4"
    source.write_bytes(b"source")
    prepared.write_bytes(b"prepared")
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "document_type": "trtvideo-compatibility-input",
                "schema_version": 1,
                "fixture_contract": FIXTURE_CONTRACT_VERSION,
                "source_kind": "user-supplied",
                "source": file_identity(source),
                "output": file_identity(prepared),
                "observed": {
                    "width": 1280,
                    "height": 720,
                    "frames": 120,
                    "fps": "24/1",
                    "audio_streams": 0,
                    "duration_sec": 5.0,
                    "timestamps": "strictly_monotonic",
                    "full_decode": True,
                },
            }
        ),
        encoding="utf-8",
    )
    prepared.write_bytes(b"changed")

    with pytest.raises(CompatibilityEvidenceError, match="does not match"):
        input_manifest_evidence(manifest, prepared)


def test_pinned_input_manifest_retains_verified_attribution(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.mp4"
    prepared.write_bytes(b"prepared")
    attribution = {
        "name": VIDEO_NAME,
        "author": VIDEO_AUTHOR,
        "source": VIDEO_SOURCE_PAGE_URL,
        "license": VIDEO_LICENSE,
        "license_url": VIDEO_LICENSE_URL,
    }
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "document_type": "trtvideo-compatibility-input",
                "schema_version": 1,
                "fixture_contract": FIXTURE_CONTRACT_VERSION,
                "source_kind": "pinned-live-action",
                "source": {
                    "name": "Jacqueville-beach-2026.webm",
                    "sha256": VIDEO_SHA256,
                    "size_bytes": VIDEO_SIZE_BYTES,
                },
                "output": file_identity(prepared),
                "observed": {
                    "width": 1280,
                    "height": 720,
                    "frames": 120,
                    "fps": "24/1",
                    "audio_streams": 1,
                    "duration_sec": 5.0,
                    "timestamps": "strictly_monotonic",
                    "full_decode": True,
                },
                "attribution": attribution,
            }
        ),
        encoding="utf-8",
    )

    evidence = input_manifest_evidence(manifest, prepared)

    assert evidence["attribution"] == attribution
