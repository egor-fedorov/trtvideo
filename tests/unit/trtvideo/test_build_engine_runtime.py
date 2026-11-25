"""TensorRT builder control flow with no native imports or GPU allocation."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trtvideo.cli import build_engine


@pytest.fixture
def native(monkeypatch, tmp_path):
    input_tensor = SimpleNamespace(name="input", shape=(1, 3, 8, 8), dtype="float32")
    output_tensor = SimpleNamespace(name="output", shape=(1, 3, 16, 16), dtype="float32")
    network = SimpleNamespace(
        get_input=Mock(return_value=input_tensor),
        get_output=Mock(return_value=output_tensor),
    )
    cache = SimpleNamespace(serialize=Mock(return_value=b"saved cache"))
    config = SimpleNamespace(
        create_timing_cache=Mock(return_value=cache),
        set_timing_cache=Mock(return_value=True),
        get_timing_cache=Mock(return_value=cache),
        set_memory_pool_limit=Mock(),
        add_optimization_profile=Mock(return_value=0),
        builder_optimization_level=0,
    )
    profile = SimpleNamespace(set_shape=Mock(return_value=None))
    builder = SimpleNamespace(
        create_network=Mock(return_value=network),
        create_builder_config=Mock(return_value=config),
        create_optimization_profile=Mock(return_value=profile),
        build_serialized_network=Mock(return_value=b"serialized engine"),
    )
    parser = SimpleNamespace(
        parse=Mock(return_value=True),
        num_errors=1,
        get_error=Mock(return_value="invalid ONNX"),
    )
    logger_factory = Mock(side_effect=lambda severity: SimpleNamespace(min_severity=severity))
    logger_factory.INFO = 3
    logger_factory.VERBOSE = 4
    logger_factory.ERROR = 1
    trt = SimpleNamespace(
        __version__="test",
        Logger=logger_factory,
        Builder=Mock(return_value=builder),
        OnnxParser=Mock(return_value=parser),
        DataType=SimpleNamespace(FLOAT="float32", HALF="float16"),
        MemoryPoolType=SimpleNamespace(WORKSPACE=0),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(build_engine, "TRT_LOGGER", None)
    monkeypatch.setattr(build_engine, "infer_onnx_precision", lambda path: "fp16")
    source = tmp_path / "model.onnx"
    source.write_bytes(b"source model")
    cache_path = tmp_path / "timing.cache"
    cache_path.write_bytes(b"previous cache")
    return SimpleNamespace(
        trt=trt,
        builder=builder,
        network=network,
        config=config,
        profile=profile,
        parser=parser,
        cache=cache,
        source=source,
        input=input_tensor,
        engine_path=tmp_path / "model.engine",
        cache_path=cache_path,
        manifest_path=tmp_path / "model.engine.json",
    )


def run_build(native, *, dynamic=False, timing_cache=True):
    shapes = {}
    if dynamic:
        native.input.shape = (-1, 3, -1, -1)
        shapes = {
            "min_shape": ("input", (1, 3, 4, 4)),
            "opt_shape": ("input", (1, 3, 8, 8)),
            "max_shape": ("input", (1, 3, 16, 16)),
        }
    return build_engine.build_engine(
        str(native.source),
        str(native.engine_path),
        **shapes,
        timing_cache_path=str(native.cache_path) if timing_cache else None,
        manifest_path=str(native.manifest_path),
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_builder_writes_engine_cache_and_unchanged_manifest_contract(native, dynamic) -> None:
    assert run_build(native, dynamic=dynamic)
    assert native.engine_path.read_bytes() == b"serialized engine"
    assert native.cache_path.read_bytes() == b"saved cache"
    native.parser.parse.assert_called_once_with(b"source model")
    native.config.create_timing_cache.assert_called_once_with(b"previous cache")
    native.config.set_timing_cache.assert_called_once_with(native.cache, False)
    native.config.set_memory_pool_limit.assert_called_once_with(0, 8 << 30)
    assert native.config.builder_optimization_level == 5
    native.builder.build_serialized_network.assert_called_once_with(native.network, native.config)
    manifest = json.loads(native.manifest_path.read_text())
    assert manifest["precision"] == "fp16"
    assert manifest["io_precision"] == "fp32"
    assert manifest["builder_flags"] == []
    assert manifest["builder_optimization_level"] == 5
    assert manifest["engine_sha256"] == build_engine.sha256_file(str(native.engine_path))
    assert manifest["model_sha256"] == build_engine.sha256_file(str(native.source))
    if dynamic:
        native.profile.set_shape.assert_called_once_with(
            "input",
            (1, 3, 4, 4),
            (1, 3, 8, 8),
            (1, 3, 16, 16),
        )
        native.config.add_optimization_profile.assert_called_once_with(native.profile)
        assert manifest["input_profile"]["opt"] == [1, 3, 8, 8]
    else:
        native.builder.create_optimization_profile.assert_not_called()
        assert manifest["input_profile"] is None


@pytest.mark.parametrize(
    ("owner", "method", "failure", "message", "compiled"),
    [
        ("builder", "create_network", None, "create TensorRT network", False),
        ("parser", "parse", False, "ONNX Parse Error", False),
        ("builder", "create_builder_config", None, "builder config", False),
        ("network", "get_input", None, "input and an output", False),
        ("network", "get_output", None, "input and an output", False),
        ("config", "create_timing_cache", None, "create TensorRT timing cache", False),
        ("config", "set_timing_cache", False, "rejected the timing cache", False),
        ("builder", "create_optimization_profile", None, "create TensorRT optimization", False),
        ("config", "add_optimization_profile", -1, "rejected the optimization profile", False),
        ("builder", "build_serialized_network", None, "Compilation failed", True),
        ("config", "get_timing_cache", None, "requested timing cache", True),
        ("cache", "serialize", None, "serialize TensorRT timing cache", True),
    ],
)
def test_builder_reports_native_failures_without_publishing_a_manifest(
    native,
    capsys,
    owner,
    method,
    failure,
    message,
    compiled,
) -> None:
    getattr(getattr(native, owner), method).return_value = failure
    assert not run_build(native, dynamic=True)
    assert message in capsys.readouterr().out
    assert not native.manifest_path.exists()
    assert native.cache_path.read_bytes() == b"previous cache"
    assert native.builder.build_serialized_network.called is compiled
    if not compiled:
        assert not native.engine_path.exists()


def test_builder_reports_native_profile_shape_rejection(native, capsys) -> None:
    native.profile.set_shape.side_effect = ValueError("fixed axis mismatch")
    assert not run_build(native, dynamic=True)
    assert "fixed axis mismatch" in capsys.readouterr().out
    native.config.add_optimization_profile.assert_not_called()
    native.builder.build_serialized_network.assert_not_called()


def test_builder_without_cache_does_not_call_cache_api(native) -> None:
    assert run_build(native, timing_cache=False)
    native.config.create_timing_cache.assert_not_called()
    native.config.set_timing_cache.assert_not_called()
    native.config.get_timing_cache.assert_not_called()


def test_builder_creates_cache_when_no_file_exists(native) -> None:
    native.cache_path.unlink()
    assert run_build(native)
    native.config.create_timing_cache.assert_called_once_with(b"")


def test_logger_is_cached_and_cli_propagates_build_failure(native, monkeypatch) -> None:
    logger = build_engine._load_tensorrt()
    assert build_engine._load_tensorrt() is logger
    native.trt.Logger.assert_called_once_with(native.trt.Logger.INFO)
    native.parser.parse.return_value = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build-engine",
            str(native.source),
            "-o",
            str(native.engine_path),
            "--quiet",
        ],
    )
    with pytest.raises(SystemExit) as error:
        build_engine.main()
    assert error.value.code == 1
    assert logger.min_severity == native.trt.Logger.ERROR
