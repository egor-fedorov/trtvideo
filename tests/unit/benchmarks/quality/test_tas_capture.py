from __future__ import annotations

import json
import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.scripts.quality import capture_tas as capture
from benchmarks.scripts.quality.model_space import (
    CaptureManifest,
    ModelSpaceError,
    create_tensor_artifact,
    write_capture_manifest,
)
from benchmarks.scripts.runtime.environment import sha256_file, write_json


class Tensor:
    """CPU-only stand-in exposing the tensor operations used for capture."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.values

    def to(self, *, device, dtype):
        assert device == "cuda"
        assert dtype == "float32"
        return self


def test_save_tensor_preserves_planar_rgb_bytes(tmp_path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    artifact = capture._save_tensor(Tensor(values), "input", 7, (3, 2, 4), tmp_path)
    assert artifact.frame_index == 7
    assert artifact.shape == (3, 2, 4)
    assert artifact.path == "input.frame-000007.f32"
    assert artifact.size_bytes == 24 * 4
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / artifact.path, dtype="<f4"), values.ravel()
    )


def test_save_tensor_rejects_hwc_even_with_matching_element_count(tmp_path) -> None:
    with pytest.raises(ModelSpaceError, match="NCHW"):
        capture._save_tensor(Tensor(np.zeros((2, 4, 3))), "input", 0, (3, 2, 4), tmp_path)


@pytest.fixture
def reader(monkeypatch):
    state = SimpleNamespace(instances=[], fallback=None)

    class Reader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.decodeMethod = kwargs["decode_method"]
            self.decodeError = None
            self.queue = Queue(maxsize=2)
            self.finished = False
            self.consumed = 0
            state.instances.append(self)

        def _decodeWithnelux(self, method):
            for index in range(4):
                self.queue.put(Tensor(np.full((1, 3, 2, 4), index, dtype=np.float32)))

        def decodeWithOpenCV(self):
            pytest.fail("OpenCV must not run")

        def __call__(self):
            try:
                if state.fallback == "opencv":
                    self.decodeWithOpenCV()
                else:
                    self._decodeWithnelux(state.fallback or self.decodeMethod)
            except Exception as exc:
                self.decodeError = exc
            finally:
                self.queue.put(None)
                self.finished = True

        def read(self):
            value = self.queue.get(timeout=2)
            if value is not None:
                self.consumed += 1
            return value

    monkeypatch.setitem(sys.modules, "src.io.ffmpegSettings", SimpleNamespace(BuildBuffer=Reader))
    return state


def test_preprocessing_captures_native_selected_frames_and_drains_reader(tmp_path, reader) -> None:
    artifacts = capture._preprocessing(tmp_path / "input.mp4", tmp_path, (1, 3), (3, 2, 4), "nvdec")
    assert [item.frame_index for item in artifacts] == [1, 3]
    instance = reader.instances[0]
    assert instance.finished is True
    assert instance.consumed == 4
    assert instance.kwargs["half"] is False
    assert instance.kwargs["toTorch"] is True
    assert instance.kwargs["decode_method"] == "nvdec"


def test_preprocessing_drains_producer_when_saving_fails(tmp_path, reader, monkeypatch) -> None:
    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(capture, "_save_tensor", fail)
    with pytest.raises(OSError, match="disk full"):
        capture._preprocessing(tmp_path / "input.mp4", tmp_path, (0,), (3, 2, 4), "nvdec")
    assert reader.instances[0].finished is True
    assert reader.instances[0].consumed == 4


@pytest.mark.parametrize("fallback", ["cpu", "opencv"])
def test_preprocessing_rejects_decoder_fallback(tmp_path, reader, fallback) -> None:
    reader.fallback = fallback
    with pytest.raises(ModelSpaceError, match="preprocessing failed"):
        capture._preprocessing(tmp_path / "input.mp4", tmp_path, (0,), (3, 2, 4), "nvdec")
    assert reader.instances[0].finished is True


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path
    input_path = root / "input.mp4"
    onnx_path = root / "model.onnx"
    engine_path = root / "model.engine"
    for path in (input_path, onnx_path, engine_path):
        path.write_bytes(path.name.encode())
    manifest = {
        "id": "test-workload",
        "clip": {"frames": 4},
        "quality": {"model_space": {"frame_indices": [0, 1]}},
    }
    metadata = root / "implementations.json"
    write_json(metadata, {"schema_version": 1, "implementations": {"tas": {"image": "test:image"}}})
    monkeypatch.setattr(capture, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        capture,
        "find_clip_variant",
        lambda *_args: {
            "path": "input.mp4",
            "benchmark_output": {"height": 4, "width": 8},
        },
    )
    monkeypatch.setattr(
        capture,
        "find_model_variant",
        lambda *_args: {
            "fp16_path": "model.onnx",
            "input_height": 2,
            "input_width": 4,
        },
    )
    monkeypatch.setattr(capture, "load_engine_contract", lambda _engine: ({}, root / "engine.json"))
    monkeypatch.setattr(capture, "validate_tas_engine_contract", lambda *_args: None)
    monkeypatch.setattr(capture, "load_upstream", lambda _gpu: None)
    image = {"id": "sha256:test", "repository_revision": "test-revision", "source_dirty": "0"}
    monkeypatch.setattr(capture, "collect_image_identity", lambda **_kwargs: image)
    canonical_dir = root / "reference"
    canonical_dir.mkdir()
    artifacts = []
    for index in (0, 1):
        for stage, shape in (("input", (3, 2, 4)), ("output", (3, 4, 8))):
            path = canonical_dir / f"{stage}-{index}.f32"
            np.full(shape, index * 0.25, dtype="<f4").tofile(path)
            artifacts.append(
                create_tensor_artifact(
                    stage=stage,
                    frame_index=index,
                    shape=shape,
                    path=path,
                    root=canonical_dir,
                )
            )
    canonical = canonical_dir / "manifest.json"
    write_capture_manifest(
        canonical,
        implementation="trtvideo",
        capture_scope="production-reference",
        workload_id=manifest["id"],
        variant="720p",
        input_sha256=sha256_file(input_path),
        onnx_sha256=sha256_file(onnx_path),
        engine_sha256=sha256_file(engine_path),
        image=image,
        execution_profile={"execution_profile": "tuned"},
        artifacts=artifacts,
    )
    args = capture.build_parser().parse_args(
        [
            "--manifest",
            str(root / "manifest.json"),
            "--implementations",
            str(metadata),
            "--variant",
            "720p",
            "--engine",
            str(engine_path),
            "--root",
            str(root),
            "--output-dir",
            str(root / "capture"),
            "--shared-input-manifest",
            str(canonical),
            "--execution-profile",
            "tuned",
            "--decode-method",
            "nvdec",
            "--writer",
            "nelux",
        ]
    )
    state = SimpleNamespace(calls=[], restored=False, reused=True, marker="original")

    class Native:
        def __init__(self, **kwargs):
            state.calls.append(kwargs)

        def __call__(self, frame, next_frame):
            assert next_frame is None
            # Distinguish actual input binding from the supplied file.
            self.dummyInput = Tensor(frame.values + 0.125)
            return Tensor(np.repeat(np.repeat(self.dummyInput.values, 2, axis=2), 2, axis=3))

    def pin(_engine, _onnx, evidence, *, patches):
        state.marker = "pinned"
        patches.callback(setattr, state, "marker", "original")
        evidence["engine_reused"] = state.reused

    monkeypatch.setattr(capture, "pin_engine", pin)
    monkeypatch.setitem(
        sys.modules, "src.upscale.tensorrt", SimpleNamespace(UniversalTensorRT=Native)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=Tensor, float32="float32"))
    return args, state, canonical


def test_shared_capture_reads_actual_binding_and_reuses_static_engine(shared) -> None:
    args, state, canonical = shared
    output = capture.capture(args)
    document = CaptureManifest.load(output)
    assert document.capture_scope == "shared-input-inference"
    assert document.canonical_input_manifest_sha256 == sha256_file(canonical)
    assert document.execution_profile == {
        "execution_profile": "tuned",
        "decode_method": "nvdec",
        "writer": "nelux",
        "cuda_graph": True,
    }
    assert state.marker == "original"
    assert len(state.calls) == 1
    assert state.calls[0]["half"] is False
    assert state.calls[0]["forceStatic"] is True
    assert state.calls[0]["width"] == 4
    assert state.calls[0]["height"] == 2
    for item in document.artifacts:
        values = np.fromfile(output.parent / item.path, dtype="<f4")
        np.testing.assert_array_equal(values, item.frame_index * 0.25 + 0.125)


@pytest.mark.parametrize(
    "mismatch", ["input-hash", "onnx-hash", "scope", "workload", "frames", "tensor-hash"]
)
def test_shared_capture_rejects_mismatched_evidence_without_success_manifest(
    shared, mismatch
) -> None:
    args, state, canonical = shared
    if mismatch == "tensor-hash":
        (canonical.parent / "input-0.f32").write_bytes(b"tampered")
    else:
        document = json.loads(canonical.read_text())
        if mismatch == "frames":
            args.frame_indices = "0"
        elif mismatch in {"input-hash", "onnx-hash"}:
            document["assets"]["input_sha256" if mismatch == "input-hash" else "onnx_sha256"] = (
                "f" * 64
            )
            write_json(canonical, document)
        else:
            document[
                {
                    "scope": "capture_scope",
                    "workload": "workload_id",
                }[mismatch]
            ] = "wrong"
            write_json(canonical, document)
    with pytest.raises(ModelSpaceError):
        capture.capture(args)
    assert not (Path(args.output_dir) / "manifest.json").exists()
    assert state.marker == "original"


def test_shared_capture_requires_engine_reuse(shared) -> None:
    args, state, _canonical = shared
    state.reused = False
    with pytest.raises(ModelSpaceError, match="did not use the prebuilt engine"):
        capture.capture(args)
    assert state.marker == "original"
    assert not (Path(args.output_dir) / "manifest.json").exists()


def test_capture_preserves_existing_outputs(shared) -> None:
    args, _state, _canonical = shared
    output = Path(args.output_dir)
    output.mkdir()
    previous = output / "manifest.json"
    previous.write_text("previous evidence")
    with pytest.raises(ModelSpaceError, match="directory is not empty"):
        capture.capture(args)
    assert previous.read_text() == "previous evidence"
