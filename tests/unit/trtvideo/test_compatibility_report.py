from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import trtvideo.compatibility.report as report_module
from trtvideo.compatibility.evidence import (
    CompatibilityEvidenceError,
    command_evidence,
    conformance_evidence,
    engine_evidence,
)
from trtvideo.compatibility.input import FIXTURE_CONTRACT_VERSION
from trtvideo.compatibility.media import MediaInspection, validate_media
from trtvideo.compatibility.report import CompatibilityRequest, generate_compatibility_report
from trtvideo.diagnostics.doctor import CheckResult, DoctorReport
from trtvideo.models.export_conformance import EXPORT_PROBE_SHA256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_engine(tmp_path: Path) -> tuple[Path, dict]:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    sidecar = {
        "schema_version": 1,
        "engine_sha256": _sha(b"engine"),
        "model_sha256": "a" * 64,
        "tensorrt_version": "11.0.0",
        "precision": "fp16",
        "io_precision": "fp32",
        "input": {
            "name": "input",
            "shape": [1, 3, 720, 1280],
            "dtype": "DataType.FLOAT",
        },
        "output": {
            "name": "output",
            "shape": [1, 3, 1440, 2560],
            "dtype": "DataType.FLOAT",
        },
        "input_profile": None,
        "preprocess_version": "uint8_to_float_0_1",
        "postprocess_version": "float_0_1_to_uint8",
        "builder_optimization_level": 5,
        "builder_flags": [],
    }
    Path(f"{engine}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return engine, sidecar


def _conformance(source: Path) -> dict:
    return {
        "document_type": "model-export-conformance",
        "schema_version": 2,
        "status": "valid",
        "export_contract": "spandrel-image-upscale-v2",
        "model": {
            "name": "example",
            "scale": 2,
            "source_sha256": _sha(source.read_bytes()),
            "source_size_bytes": source.stat().st_size,
        },
        "probe": {
            "version": "rgb-coordinate-pattern-v1",
            "input_shape": [1, 3, 16, 16],
            "output_shape": [1, 3, 32, 32],
            "input_sha256": EXPORT_PROBE_SHA256,
        },
        "comparison": {
            "reference": "pytorch-fp32",
            "candidate": "onnxruntime-cpu-fp32",
            "thresholds": {"max_abs": 0.0001, "rmse": 0.00001, "min_psnr_db": 80.0},
            "metrics": {
                "max_abs": 0.00001,
                "mean_abs": 0.000001,
                "rmse": 0.000001,
                "relative_l2": 0.000001,
                "psnr_db": 100.0,
            },
        },
        "tools": {
            "torch": "test",
            "onnx": "test",
            "onnxruntime": "test",
            "onnxscript": "test",
            "spandrel": "test",
        },
        "exports": [{"path": "model.onnx", "sha256": "c" * 64, "size_bytes": 10}],
    }


def _stream(*, width: int, height: int, frames: int, codec: str) -> dict:
    return {
        "codec_type": "video",
        "codec_name": codec,
        "width": width,
        "height": height,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "avg_frame_rate": "24/1",
        "nb_read_frames": str(frames),
        "duration": str(frames / 24),
        "bit_rate": "4000000",
        "has_b_frames": 0,
    }


def _inspection(
    *,
    name: str,
    width: int,
    height: int,
    frames: int,
    codec: str,
    packets: bool,
) -> MediaInspection:
    return MediaInspection(
        identity={"name": name, "sha256": "d" * 64, "size_bytes": 100},
        probe={
            "streams": [
                _stream(width=width, height=height, frames=frames, codec=codec),
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "codec_tag_string": "mp4a",
                    "tags": {"language": "eng"},
                },
            ],
            "format": {"duration": str(frames / 24), "bit_rate": "4200000"},
            "chapters": [],
        },
        packets=(
            tuple(
                {"pts": index, "dts": index, "flags": "K_" if index == 0 else "__"}
                for index in range(frames)
            )
            if packets
            else ()
        ),
        decode_error=None,
    )


def test_engine_evidence_rejects_changed_engine(tmp_path: Path) -> None:
    engine, _sidecar = _write_engine(tmp_path)
    engine.write_bytes(b"changed")

    with pytest.raises(CompatibilityEvidenceError, match="hash does not match"):
        engine_evidence(engine)


def test_command_evidence_rejects_private_home_path(tmp_path: Path) -> None:
    commands = tmp_path / "commands.txt"
    commands.write_text("trtvideo --engine /home/alice/model.engine", encoding="utf-8")

    with pytest.raises(CompatibilityEvidenceError, match="user-home path"):
        command_evidence(commands)


def test_conformance_evidence_rejects_self_declared_thresholds(tmp_path: Path) -> None:
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    report = _conformance(source)
    report["comparison"]["thresholds"]["max_abs"] = 1.0
    report_path = tmp_path / "model.export-conformance.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(CompatibilityEvidenceError, match="thresholds are not canonical"):
        conformance_evidence(
            report_path,
            source_identity={
                "name": source.name,
                "sha256": _sha(source.read_bytes()),
                "size_bytes": source.stat().st_size,
            },
            expected_scale=2,
        )


def test_media_validation_binds_video_to_engine_contract(tmp_path: Path) -> None:
    engine, _sidecar = _write_engine(tmp_path)
    _summary, model = engine_evidence(engine)

    result = validate_media(
        _inspection(
            name="input.mp4",
            width=1280,
            height=720,
            frames=200,
            codec="h264",
            packets=False,
        ),
        _inspection(
            name="output.mp4",
            width=2560,
            height=1440,
            frames=120,
            codec="h264",
            packets=True,
        ),
        model=model,
        expected_frames=120,
    )

    assert result["valid"]
    assert result["output"]["pts_monotonic"]
    assert result["output"]["full_decode"]
    assert result["output"]["video"]["video_bitrate_bps"] == 4_000_000


def test_generate_report_writes_sanitized_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    conformance_path = tmp_path / "model.export-conformance.json"
    conformance_path.write_text(json.dumps(_conformance(source)), encoding="utf-8")
    engine, _sidecar = _write_engine(tmp_path)
    commands = tmp_path / "commands.txt"
    commands.write_text(
        "export-onnx models/model.pth --output_dir models/onnx\n"
        "trtvideo --engine models/model.engine --input videos/input.mp4",
        encoding="utf-8",
    )
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"input")
    output_video.write_bytes(b"output")
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(
        json.dumps(
            {
                "document_type": "trtvideo-compatibility-input",
                "schema_version": 1,
                "fixture_contract": FIXTURE_CONTRACT_VERSION,
                "source_kind": "user-supplied",
                "source": {
                    "name": "source.mp4",
                    "sha256": "f" * 64,
                    "size_bytes": 200,
                },
                "output": {
                    "name": "input.mp4",
                    "sha256": _sha(b"input"),
                    "size_bytes": len(b"input"),
                },
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
    input_inspection = _inspection(
        name="input.mp4",
        width=1280,
        height=720,
        frames=200,
        codec="h264",
        packets=False,
    )
    output_inspection = _inspection(
        name="output.mp4",
        width=2560,
        height=1440,
        frames=120,
        codec="h264",
        packets=True,
    )
    monkeypatch.setattr(
        report_module,
        "inspect_media",
        lambda path, **_kwargs: input_inspection if path == input_video else output_inspection,
    )
    output_dir = tmp_path / "report"

    def fake_doctor(*, disk_path: Path, **_kwargs) -> DoctorReport:
        assert disk_path == output_dir
        assert disk_path.is_dir()
        return DoctorReport(
            checks=(
                CheckResult("GPU", True, "Test GPU"),
                CheckResult("Disk", True, "10.00 GiB free of 20.00 GiB at /private/path"),
            )
        )

    monkeypatch.setattr(report_module, "run_doctor", fake_doctor)
    monkeypatch.setenv("TRTVIDEO_BUILD_REVISION", "e" * 40)
    monkeypatch.setenv("TRTVIDEO_BUILD_DIRTY", "0")
    monkeypatch.setenv("TRTVIDEO_BASE_IMAGE", "nvcr.io/nvidia/tensorrt:test")

    report = generate_compatibility_report(
        CompatibilityRequest(
            model_name="2xExample",
            model_source="https://example.test/model",
            model_license="MIT",
            source_format="checkpoint",
            source_artifact=source,
            engine=engine,
            input_video=input_video,
            output_video=output_video,
            expected_frames=120,
            commands_file=commands,
            export_conformance=conformance_path,
            image_reference="ghcr.io/example/trtvideo:test",
            gpu_id=0,
            output_dir=output_dir,
            input_manifest=input_manifest,
        )
    )

    assert report["status"] == "valid"
    json_text = report["written_files"]["json"].read_text(encoding="utf-8")
    markdown = report["written_files"]["markdown"].read_text(encoding="utf-8")
    assert str(tmp_path) not in json_text
    assert "/private/path" not in json_text
    assert "2xExample" in markdown
    assert "community-reported" in markdown
    assert report["input_preparation"]["prepared_input"]["sha256"] == _sha(b"input")
