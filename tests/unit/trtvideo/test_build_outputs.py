import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trtvideo.cli import build_engine
from trtvideo.models.build_outputs import validate_destinations


@pytest.fixture
def build(tmp_path, monkeypatch, make_weighted_onnx):
    source = make_weighted_onnx(tmp_path / "source.onnx")
    trt = MagicMock(__version__="test")
    trt.DataType.FLOAT = "float"
    builder = trt.Builder.return_value
    network = builder.create_network.return_value
    tensor = SimpleNamespace(name="input", shape=(1, 3, 4, 4), dtype="float")
    network.get_input.return_value = network.get_output.return_value = tensor
    config = builder.create_builder_config.return_value
    builder.build_serialized_network.return_value = b"new engine"
    config.get_timing_cache.return_value.serialize.return_value = b"new cache"
    monkeypatch.setattr(build_engine, "trt", trt)
    paths = [tmp_path / "engine", tmp_path / "cache", tmp_path / "subdir" / "manifest"]

    def run():
        return build_engine.build_engine(
            str(source), str(paths[0]), timing_cache_path=str(paths[1]), manifest_path=str(paths[2])
        )

    return run, paths, config


def test_build_publishes_final_paths_and_hash(build):
    run, (engine, cache, manifest), _ = build
    assert run()
    record = json.loads(manifest.read_text())
    assert record["engine_sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert record["engine_path"] == str(engine)
    assert record["timing_cache"] == str(cache)
    assert cache.read_bytes() == b"new cache"


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("failure", ["create", "attach", "get", "serialize", "cache", "manifest"])
def test_failed_build_preserves_complete_set(build, previous, failure, monkeypatch, tmp_path):
    run, paths, config = build
    original = os.replace
    if previous:
        for path in paths:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(path.name.encode())

    def replace(src, dest):
        if Path(src).name == "new" and Path(dest).name == failure:
            raise OSError("injected rename failure")
        original(src, dest)

    monkeypatch.setattr(os, "replace", replace)
    if failure == "create":
        config.create_timing_cache.return_value = None
    elif failure == "attach":
        config.set_timing_cache.return_value = False
    elif failure == "get":
        config.get_timing_cache.return_value = None
    elif failure == "serialize":
        config.get_timing_cache.return_value.serialize.return_value = None
    with pytest.raises((OSError, ValueError), match="timing cache|rename failure"):
        run()
    assert not list(tmp_path.rglob(".build-*"))
    for path in paths:
        assert path.exists() == previous
        if previous:
            assert path.read_bytes() == path.name.encode()


def test_build_rejects_overlapping_outputs_before_builder(
    tmp_path, make_weighted_onnx, monkeypatch
):
    graph = make_weighted_onnx(tmp_path / "source.onnx")
    monkeypatch.setattr(build_engine, "_load_tensorrt", lambda: pytest.fail("builder started"))
    with pytest.raises(ValueError, match="overlapping"):
        build_engine.build_engine(str(graph), str(graph), timing_cache_path=str(graph))


@pytest.mark.parametrize("reverse", [False, True])
def test_output_must_not_be_another_artifacts_parent(tmp_path, reverse):
    engine = tmp_path / "model.engine"
    paths = [engine, engine / "cache"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_destinations(paths[::-1] if reverse else paths, [])
    assert not engine.exists()
