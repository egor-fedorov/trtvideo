from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scripts.runners import tas_process as process
from benchmarks.scripts.runners.tas_runtime import patch_attribute
from benchmarks.scripts.runtime.environment import write_json
from trtvideo.benchmarking.lifecycle import load_frame_markers
from trtvideo.video.nvcodec.encoder import NvencCbrContract


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    args = process.build_parser().parse_args(
        [
            "--engine",
            str(tmp_path / "model.engine"),
            "--onnx",
            str(tmp_path / "model.onnx"),
            "--input",
            str(tmp_path / "input.mp4"),
            "--output",
            str(tmp_path / "output.mp4"),
            "--encoder-contract",
            str(tmp_path / "encoder.json"),
            "--frames",
            "2",
            "--fps",
            "24/1",
            "--decode-method",
            "nvdec",
            "--writer",
            "nelux",
            "--lifecycle-output",
            str(tmp_path / "lifecycle.json"),
            "--runtime-evidence",
            str(tmp_path / "runtime.json"),
        ]
    )
    write_json(args.encoder_contract, NvencCbrContract(35_000_000, 24).as_dict())
    write_json(Path(f"{args.engine}.json"), {"engine_sha256": "e" * 64, "model_sha256": "c" * 64})
    state = SimpleNamespace(
        scenario="valid",
        args=args,
        reused=True,
        submitted=[],
        decoded=[],
        inputs=[],
        models=[],
        callbacks=[],
        native_argv=None,
        marker="original",
    )

    class Reader:
        def __init__(self):
            self.decodeMethod = args.decode_method
            self.decodeError = None

        def _decodeWithnelux(self, method):
            state.decoded.append(method)
            return args.frames

        def decodeWithOpenCV(self):
            pytest.fail("native OpenCV fallback must never be executed")

    class NeluxWriter:
        encodeError = None

    class FfmpegWriter:
        encodeError = None

    class Processor:
        def _writeOut(self, frame):
            state.submitted.append(frame)

    class Model:
        def __init__(self):
            self.dummyInput = object()
            state.models.append(self)

        def __call__(self, frame, next_frame):
            assert next_frame is None
            state.inputs.append(frame)
            return object()

    def writer():
        if state.scenario == "unknown-writer":
            return object()
        instance = FfmpegWriter() if state.scenario == "writer-fallback" else NeluxWriter()
        if state.scenario == "encoder-error":
            instance.encodeError = RuntimeError("NVENC failed")
        return instance

    buffers = SimpleNamespace(
        BuildBuffer=Reader,
        NeluxWriteBuffer=NeluxWriter,
        WriteBuffer=FfmpegWriter,
        createWriteBuffer=writer,
    )

    def main():
        state.native_argv = sys.argv.copy()
        if state.scenario == "exit-code":
            raise SystemExit(7)
        reader = Reader()
        reader._decodeWithnelux(args.decode_method)
        if state.scenario == "decoder-fallback":
            with pytest.raises(RuntimeError, match="fallback"):
                reader._decodeWithnelux("cpu")
        if state.scenario == "opencv-fallback":
            with pytest.raises(RuntimeError, match="OpenCV"):
                reader.decodeWithOpenCV()
        if state.scenario == "decoder-drift":
            reader.decodeMethod = "cpu"
        if state.scenario == "decoder-error":
            reader.decodeError = RuntimeError("decode failed")
        buffers.createWriteBuffer()
        native = Model()
        for index in range(args.frames):
            if state.scenario == "short-output" and index == 1:
                break
            output = native(object(), None)
            Processor()._writeOut(output)
            if state.scenario == "main-error":
                raise RuntimeError("upstream processing failed")
        if state.scenario != "missing-output":
            args.output.write_bytes(b"encoded")

    native_module = SimpleNamespace(VideoProcessor=Processor, main=main)
    dependencies = SimpleNamespace(
        DependencyChecker=type("Checker", (), {"ensureDependencies": lambda _self: False}),
        installDependencies=lambda: pytest.fail("installation must never be executed"),
    )
    ffmpeg = SimpleNamespace(getFFMPEG=lambda: pytest.fail("download must never be executed"))
    for name, module in {
        "src.io.ffmpegSettings": buffers,
        "src.upscale.tensorrt": SimpleNamespace(UniversalTensorRT=Model),
        "src.infra.dependencyHandler": dependencies,
        "src.infra.getFFMPEG": ffmpeg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(process, "load_upstream", lambda _gpu: native_module)
    monkeypatch.setattr(process, "adapter_sha256", lambda: "a" * 64)
    monkeypatch.setattr(process, "runtime_identity", lambda: {"tensorrt": "test"})
    monkeypatch.setattr(process.importlib.metadata, "version", lambda _name: "test")

    def pin(_engine, _onnx, evidence, *, patches):
        evidence["engine_reused"] = state.reused
        patch_attribute(patches, state, "marker", "pinned")

    monkeypatch.setattr(process, "pin_engine", pin)
    monkeypatch.setattr(process, "configure_encoders", lambda _contract, *, patches: None)
    saved = [
        (owner, name, getattr(owner, name))
        for owner, name in (
            (Reader, "_decodeWithnelux"),
            (Reader, "decodeWithOpenCV"),
            (buffers, "createWriteBuffer"),
            (Processor, "_writeOut"),
            (Model, "__call__"),
            (dependencies.DependencyChecker, "ensureDependencies"),
            (dependencies, "installDependencies"),
            (ffmpeg, "getFFMPEG"),
            (sys, "argv"),
        )
    ]

    def assert_restored():
        for owner, name, value in saved:
            assert getattr(owner, name) is value
        assert state.marker == "original"

    state.assert_restored = assert_restored
    state.dependencies = dependencies
    state.ffmpeg = ffmpeg
    return state


def test_native_arguments_keep_full_video_static_fp32_execution(upstream) -> None:
    args = process.native_arguments(upstream.args)
    assert args[args.index("--half") + 1] == "false"
    assert "--static" in args
    assert args[args.index("--custom_model") + 1] == str(upstream.args.onnx)
    assert args[args.index("--encode_method") + 1] == "nvenc_h264_nelux"
    assert args[args.index("--decode_method") + 1] == "nvdec"
    assert float(args[args.index("--outpoint") + 1]) == pytest.approx(2 / 24)
    assert "--benchmark" not in args
    upstream.args.writer = "ffmpeg"
    args = process.native_arguments(upstream.args)
    assert args[args.index("--encode_method") + 1] == "nvenc_h264"


def test_tas_run_captures_actual_binding_and_restores_callbacks_between_invocations(
    upstream,
) -> None:
    observed = []
    for _attempt in range(2):
        evidence = process.run(upstream.args, on_tensor=lambda *values: observed.append(values))
        assert evidence["status"] == "valid"
        assert evidence["errors"] == []
        assert evidence["engine_reused"] is True
        assert evidence["processed_frames"] == 2
        assert evidence["decode_method"] == "nvdec"
        assert evidence["writer"] == "nelux"
        upstream.assert_restored()
    assert [item[0] for item in observed] == [0, 1, 0, 1]
    for index, (_, bound_input, output) in enumerate(observed):
        assert bound_input is upstream.models[index // 2].dummyInput
        assert bound_input is not upstream.inputs[index]
        assert output is upstream.submitted[index]
    assert json.loads(upstream.args.runtime_evidence.read_text()) == evidence
    lifecycle = load_frame_markers(upstream.args.lifecycle_output)
    assert lifecycle.processed_frames == 2
    assert lifecycle.instrumentation == "tas-output-queue-submission"


@pytest.mark.parametrize(
    "scenario,match",
    [
        ("decoder-fallback", "decoder fallback"),
        ("opencv-fallback", "decoder fallback"),
        ("decoder-drift", "changed backend"),
        ("decoder-error", "decoder failed"),
        ("writer-fallback", "writer fallback"),
        ("unknown-writer", "unsupported output writer"),
        ("encoder-error", "NVENC failed"),
        ("short-output", "processed 1/2"),
        ("missing-output", "no output"),
        ("main-error", "upstream processing failed"),
        ("exit-code", "code 7"),
    ],
)
def test_tas_process_rejects_fallback_and_persists_failure(upstream, scenario, match) -> None:
    upstream.scenario = scenario
    with pytest.raises(RuntimeError, match=match):
        process.run(upstream.args, on_tensor=lambda *_args: None)
    evidence = json.loads(upstream.args.runtime_evidence.read_text())
    assert evidence["status"] == "invalid"
    assert evidence["errors"]
    assert not upstream.args.lifecycle_output.exists()
    upstream.assert_restored()


def test_successful_media_without_reused_engine_is_invalid(upstream) -> None:
    upstream.reused = False
    with pytest.raises(RuntimeError, match="did not load engine"):
        process.run(upstream.args)
    assert json.loads(upstream.args.runtime_evidence.read_text())["engine_reused"] is False
    upstream.assert_restored()


@pytest.mark.parametrize("stage", ["contract", "sidecar", "load", "encoder", "observer"])
def test_tas_records_setup_and_capture_errors_and_restores_partial_patches(
    upstream, monkeypatch, stage
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate failure")

    callback = None
    if stage == "contract":
        upstream.args.encoder_contract.write_text("invalid JSON")
    elif stage == "sidecar":
        Path(f"{upstream.args.engine}.json").unlink()
    elif stage == "load":
        monkeypatch.setattr(process, "load_upstream", fail)
    elif stage == "encoder":
        monkeypatch.setattr(process, "configure_encoders", fail)
    elif stage == "observer":
        callback = fail
    with pytest.raises((RuntimeError, ValueError, OSError)):
        process.run(upstream.args, on_tensor=callback)
    evidence = json.loads(upstream.args.runtime_evidence.read_text())
    assert evidence["status"] == "invalid"
    assert evidence["errors"]
    upstream.assert_restored()


def test_hermetic_dependencies_are_checked_and_restored(upstream) -> None:
    from contextlib import ExitStack

    with ExitStack() as patches:
        process._hermetic_dependencies(patches)
        assert upstream.dependencies.DependencyChecker().ensureDependencies() is True
        with pytest.raises(RuntimeError, match="forbidden"):
            upstream.dependencies.installDependencies()
        with pytest.raises(RuntimeError, match="forbidden"):
            upstream.ffmpeg.getFFMPEG()
    upstream.assert_restored()
