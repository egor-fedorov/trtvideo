from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
    validate_static_engine_contract,
    validate_vsgan_engine_contract,
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


def test_vsgan_engine_rejects_different_base_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"canonical onnx")
    sidecar = {
        "model_sha256": hashlib.sha256(b"canonical onnx").hexdigest(),
        "io_precision": "fp32",
        "input_profile": None,
        "input": {"shape": [1, 3, 1080, 1920]},
        "output": {"shape": [1, 3, 2160, 3840]},
        "builder_flags": ["stronglyTyped"],
        "tensorrt_version": "101600",
        "builder_base_image": "old-image@sha256:old",
    }
    monkeypatch.setenv("TRTVIDEO_BASE_IMAGE", "new-image@sha256:new")
    monkeypatch.setenv("TRTVIDEO_VSGAN_FFMPEG_PACKAGE", "ffmpeg-version")

    with pytest.raises(EngineContractError, match="different base image"):
        validate_vsgan_engine_contract(
            sidecar,
            manifest(),
            "1080p",
            onnx_path,
            "new-image@sha256:new",
            "ffmpeg-version",
        )
