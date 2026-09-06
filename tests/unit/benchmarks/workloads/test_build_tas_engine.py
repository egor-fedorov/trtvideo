from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.scripts.contracts.benchmark import CompetitorError
from benchmarks.scripts.contracts.engine import EngineContractError, load_engine_contract
from benchmarks.scripts.workloads.build_tas_engine import build_parser, build_tas_engine


@pytest.fixture
def native_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, Mock]:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"canonical onnx")
    engine = tmp_path / "engines" / "model.engine"
    manifest = json.loads(
        Path("benchmarks/workloads/liveaction_span_madrid.json").read_text(encoding="utf-8")
    )
    manifest["model"]["variants"][0]["fp16_path"] = str(onnx)
    manifest_path = tmp_path / "workload.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata = json.loads(Path("benchmarks/implementations.json").read_text())["implementations"][
        "tas"
    ]
    sidecar = {
        "schema_version": 1,
        "engine_sha256": hashlib.sha256(b"native engine").hexdigest(),
        "model_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        "tensorrt_version": metadata["tensorrt_version"],
        "input": {"name": "input", "shape": [1, 3, 720, 1280], "dtype": "float32"},
        "output": {"name": "output", "shape": [1, 3, 1440, 2560], "dtype": "float32"},
        "io_precision": "fp32",
        "input_profile": None,
        "builder_flags": ["stronglyTyped"],
        "builder": "tas-native",
        "tas_revision": metadata["source_revision"],
        "adapter_sha256": "a" * 64,
        "runtime": {
            "torch": metadata["torch_version"],
            "tensorrt": metadata["tensorrt_version"],
            "nelux": metadata["nelux_version"],
            "ffmpeg": "ffmpeg version 6.1.1",
        },
        "builder_base_image": "ubuntu:24.04",
    }

    def create_engine(**kwargs: object) -> dict:
        assert kwargs["engine"] == engine
        engine.write_bytes(b"native engine")
        return sidecar

    builder = Mock(side_effect=create_engine)
    monkeypatch.setitem(
        sys.modules,
        "benchmarks.scripts.runners.tas_runtime",
        SimpleNamespace(build_engine=builder, adapter_sha256=lambda: "a" * 64),
    )
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", metadata["source_revision"])
    monkeypatch.setenv("TRTVIDEO_BASE_IMAGE", "ubuntu:24.04")
    arguments = {
        "manifest_path": manifest_path,
        "variant_name": "720p",
        "onnx_path": onnx,
        "engine_path": engine,
        "gpu_id": 2,
    }
    return arguments, sidecar, builder


def test_native_builder_preserves_observed_sidecar(native_build: tuple[dict, dict, Mock]) -> None:
    arguments, sidecar, builder = native_build

    result = build_tas_engine(**arguments)

    builder.assert_called_once_with(
        onnx=arguments["onnx_path"],
        engine=arguments["engine_path"],
        width=1280,
        height=720,
        gpu_id=2,
    )
    loaded, sidecar_path = load_engine_contract(arguments["engine_path"])
    assert result == sidecar_path
    assert loaded == sidecar
    assert loaded["engine_path"] == str(arguments["engine_path"])
    assert loaded["onnx_path"] == str(arguments["onnx_path"])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("engine_sha256", "0" * 64, "engine SHA256"),
        ("model_sha256", "0" * 64, "canonical ONNX"),
        ("schema_version", 99, "unsupported sidecar schema"),
        ("adapter_sha256", "b" * 64, "adapter SHA256 changed"),
        ("output", {"shape": [1, 3, 720, 1280], "dtype": "float32"}, "output shape"),
        ("io_precision", "fp16", "FP32"),
    ],
)
def test_native_builder_does_not_publish_bad_sidecar(
    native_build: tuple[dict, dict, Mock], key: str, value: object, message: str
) -> None:
    arguments, sidecar, _ = native_build
    sidecar[key] = value
    with pytest.raises(EngineContractError, match=message):
        build_tas_engine(**arguments)
    assert not Path(f"{arguments['engine_path']}.json").exists()


def test_native_builder_requires_engine_file(native_build: tuple[dict, dict, Mock]) -> None:
    arguments, sidecar, builder = native_build
    builder.side_effect = None
    builder.return_value = sidecar
    with pytest.raises(EngineContractError, match="without creating an engine"):
        build_tas_engine(**arguments)
    assert not Path(f"{arguments['engine_path']}.json").exists()


def test_native_builder_requires_canonical_path(native_build: tuple[dict, dict, Mock]) -> None:
    arguments, _, builder = native_build
    arguments["onnx_path"] = arguments["onnx_path"].with_name("another.onnx")
    with pytest.raises(CompetitorError, match="Expected canonical ONNX"):
        build_tas_engine(**arguments)
    builder.assert_not_called()


def test_native_builder_requires_existing_onnx(native_build: tuple[dict, dict, Mock]) -> None:
    arguments, _, builder = native_build
    arguments["onnx_path"].unlink()
    with pytest.raises(CompetitorError, match="ONNX not found"):
        build_tas_engine(**arguments)
    builder.assert_not_called()


def test_native_builder_requires_valid_gpu_id(native_build: tuple[dict, dict, Mock]) -> None:
    arguments, _, builder = native_build
    arguments["gpu_id"] = -1
    with pytest.raises(CompetitorError, match="GPU id"):
        build_tas_engine(**arguments)
    builder.assert_not_called()


def test_native_builder_cli() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "workload.json",
            "--variant",
            "720p",
            "--onnx",
            "model.onnx",
            "--output",
            "model.engine",
            "--gpu-id",
            "2",
        ]
    )
    assert args.output == "model.engine"
    assert args.gpu_id == 2
