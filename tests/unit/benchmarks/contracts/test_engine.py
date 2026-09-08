from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
    validate_static_engine_contract,
    validate_tas_engine_contract,
)

MANIFEST_PATH = Path("benchmarks/workloads/realesrgan_x2plus_madrid.json")


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_load_engine_contract_verifies_engine_hash(tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    sidecar = {
        "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
        "input": {"shape": [1, 3, 720, 1280]},
        "output": {"shape": [1, 3, 1440, 2560]},
    }
    Path(f"{engine}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    loaded, sidecar_path = load_engine_contract(engine)

    assert loaded == sidecar
    assert sidecar_path == Path(f"{engine}.json")


def test_static_engine_contract_checks_onnx_and_bindings(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"canonical onnx")
    sidecar = {
        "model_sha256": hashlib.sha256(b"canonical onnx").hexdigest(),
        "io_precision": "fp32",
        "input_profile": None,
        "input": {"shape": [1, 3, 1080, 1920]},
        "output": {"shape": [1, 3, 2160, 3840]},
    }

    validate_static_engine_contract(sidecar, manifest(), "1080p", onnx_path)

    sidecar["io_precision"] = "fp16"
    with pytest.raises(EngineContractError, match="FP32"):
        validate_static_engine_contract(sidecar, manifest(), "1080p", onnx_path)


@pytest.fixture
def tas_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, dict]:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"canonical onnx")
    sidecar = {
        "model_sha256": hashlib.sha256(b"canonical onnx").hexdigest(),
        "io_precision": "fp32",
        "input_profile": None,
        "input": {"shape": [1, 3, 1080, 1920], "dtype": "float32"},
        "output": {"shape": [1, 3, 2160, 3840], "dtype": "float32"},
        "builder_flags": ["stronglyTyped"],
        "builder": "tas-native",
        "tensorrt_version": "11.2.1.2",
        "builder_base_image": "ubuntu:24.04",
        "tas_revision": "ac259ddf13c191230a4c65a4b251c9fb28884104",
        "adapter_sha256": "a" * 64,
        "runtime": {
            "tensorrt": "11.2.1.2",
            "torch": "2.13.0+cu130",
            "nelux": "0.18.0",
            "ffmpeg": "ffmpeg version 6.1.1",
        },
    }
    metadata = json.loads(Path("benchmarks/implementations.json").read_text())["implementations"][
        "tas"
    ]
    monkeypatch.setenv("TRTVIDEO_BASE_IMAGE", "ubuntu:24.04")
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", metadata["source_revision"])
    monkeypatch.setattr("benchmarks.scripts.runners.tas_runtime.adapter_sha256", lambda: "a" * 64)
    return onnx_path, sidecar, metadata


def test_tas_engine_accepts_pinned_native_builder(tas_contract: tuple[Path, dict, dict]) -> None:
    onnx_path, sidecar, metadata = tas_contract
    validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("builder", "trtexec", "native TAS builder"),
        ("builder_flags", [], "strongly typed"),
        ("tas_revision", "other-revision", "upstream revision"),
        ("tensorrt_version", "11.0.0", "contradicts"),
        ("runtime", None, "runtime provenance"),
        ("adapter_sha256", "unknown", "adapter SHA256"),
        ("adapter_sha256", "b" * 64, "adapter SHA256 changed; rebuild the engine"),
        ("builder_base_image", "old-image", "different or unknown base image"),
        ("builder_base_image", "unknown", "different or unknown base image"),
        ("input_profile", {"min": [1, 3, 720, 1280]}, "static full-frame"),
        ("model_sha256", "0" * 64, "canonical ONNX"),
        ("io_precision", "fp16", "FP32"),
    ],
)
def test_tas_engine_rejects_changed_contract(
    tas_contract: tuple[Path, dict, dict], key: str, value: object, message: str
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    sidecar[key] = value
    with pytest.raises(EngineContractError, match=message):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


@pytest.mark.parametrize("package", ["tensorrt", "torch", "nelux"])
def test_tas_engine_rejects_changed_dependency(
    tas_contract: tuple[Path, dict, dict], package: str
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    sidecar["runtime"][package] = "other-version"
    with pytest.raises(EngineContractError, match=f"{package} version"):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


def test_tas_engine_rechecks_current_adapter(
    tas_contract: tuple[Path, dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)

    monkeypatch.setattr("benchmarks.scripts.runners.tas_runtime.adapter_sha256", lambda: "b" * 64)
    with pytest.raises(EngineContractError, match="adapter SHA256 changed"):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


@pytest.mark.parametrize("tensor", ["input", "output"])
@pytest.mark.parametrize("field", ["shape", "dtype"])
def test_tas_engine_rejects_changed_bindings(
    tas_contract: tuple[Path, dict, dict], tensor: str, field: str
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    sidecar[tensor][field] = [1, 3, 720, 1280] if field == "shape" else "float16"
    with pytest.raises(EngineContractError, match="shape|bindings"):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


def test_tas_engine_rejects_wrong_runtime_revision(
    tas_contract: tuple[Path, dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", "other-revision")
    with pytest.raises(EngineContractError, match="runtime does not match"):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)


@pytest.mark.parametrize("ffmpeg", [None, "", "unknown"])
def test_tas_engine_requires_encoder_provenance(
    tas_contract: tuple[Path, dict, dict], ffmpeg: object
) -> None:
    onnx_path, sidecar, metadata = tas_contract
    sidecar["runtime"]["ffmpeg"] = ffmpeg
    with pytest.raises(EngineContractError, match="FFmpeg provenance"):
        validate_tas_engine_contract(sidecar, manifest(), "1080p", onnx_path, metadata)
