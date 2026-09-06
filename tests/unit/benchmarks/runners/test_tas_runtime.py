from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scripts.runners import tas_runtime as runtime
from benchmarks.scripts.runtime.environment import sha256_file, write_json
from trtvideo.video.nvcodec.encoder import NvencCbrContract


class Engine:
    num_io_tensors = 2

    def __init__(self):
        self.shapes = {"input": [1, 3, 4, 6], "output": [1, 3, 8, 12]}
        self.dtypes = {"input": "float32", "output": "float32"}
        self.modes = {"input": "INPUT", "output": "OUTPUT"}

    def get_tensor_name(self, index):
        return ("input", "output")[index]

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def get_tensor_dtype(self, name):
        return self.dtypes[name]

    def get_tensor_mode(self, name):
        return self.modes[name]


@pytest.fixture
def native(monkeypatch, tmp_path):
    engine = Engine()
    context = object()
    calls = []

    def load(path):
        calls.append(path)
        return engine, context

    handler = SimpleNamespace(
        tensorRTEngineLoader=load,
        tensorRTEngineNameHandler=lambda **_kwargs: "native.engine",
        tensorRTEngineCreator=lambda **_kwargs: pytest.fail("native builder must not run"),
    )

    class Model:
        def _detectRequiredMultiple(self):
            return 4

    monkeypatch.setitem(sys.modules, "src.model.trtHandler", handler)
    monkeypatch.setitem(
        sys.modules, "src.upscale.tensorrt", SimpleNamespace(UniversalTensorRT=Model)
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(
            float32="float32", TensorIOMode=SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT")
        ),
    )
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    write_json(Path(f"{path}.json"), {"input": {"shape": [1, 3, 4, 6]}})
    return SimpleNamespace(
        handler=handler,
        model=Model,
        engine=engine,
        context=context,
        calls=calls,
        path=path,
        onnx=onnx,
    )


def test_pin_engine_reuses_loader_and_restores_every_override(native) -> None:
    original = vars(native.handler).copy()
    original_shape = native.model._detectRequiredMultiple
    evidence = {}
    with ExitStack() as patches:
        runtime.pin_engine(native.path, native.onnx, evidence, patches=patches)
        assert native.handler.tensorRTEngineNameHandler(modelPath=str(native.onnx)) == str(
            native.path
        )
        assert native.handler.tensorRTEngineNameHandler(str(native.onnx)) == str(native.path)
        assert native.handler.tensorRTEngineLoader(str(native.path)) == (
            native.engine,
            native.context,
        )
        model = native.model()
        model.width, model.height, model.modelPath = 6, 4, str(native.onnx)
        assert model._detectRequiredMultiple() == 1
        assert evidence == {"engine_reused": True}
    assert vars(native.handler) == original
    assert native.model._detectRequiredMultiple is original_shape
    assert native.calls == [str(native.path)]


@pytest.mark.parametrize("action", ["build", "onnx", "engine", "shape", "shape-model", "missing"])
def test_pin_engine_forbids_build_and_contract_fallback(native, action) -> None:
    with ExitStack() as patches:
        runtime.pin_engine(native.path, native.onnx, {}, patches=patches)
        with pytest.raises(RuntimeError):
            if action == "build":
                native.handler.tensorRTEngineCreator()
            elif action == "onnx":
                native.handler.tensorRTEngineNameHandler(modelPath="different.onnx")
            elif action == "engine":
                native.handler.tensorRTEngineLoader("different.engine")
            elif action == "missing":
                native.path.unlink()
                native.handler.tensorRTEngineLoader(str(native.path))
            else:
                model = native.model()
                model.width, model.height = (7, 4) if action == "shape" else (6, 4)
                model.modelPath = str(native.onnx) if action == "shape" else "different.onnx"
                model._detectRequiredMultiple()


def test_pin_engine_loader_failure_does_not_record_reuse(native) -> None:
    native.handler.tensorRTEngineLoader = lambda _path: (None, None)
    evidence = {}
    with ExitStack() as patches:
        runtime.pin_engine(native.path, native.onnx, evidence, patches=patches)
        with pytest.raises(RuntimeError, match="could not reuse"):
            native.handler.tensorRTEngineLoader(str(native.path))
    assert evidence == {}


@pytest.mark.parametrize("mismatch", ["count", "dtype", "mode", "dynamic", "channels", "batch"])
def test_native_engine_metadata_rejects_noncanonical_bindings(native, mismatch) -> None:
    engine = native.engine
    if mismatch == "count":
        engine.num_io_tensors = 3
    elif mismatch == "dtype":
        engine.dtypes["output"] = "float16"
    elif mismatch == "mode":
        engine.modes["input"] = "OUTPUT"
    else:
        engine.shapes["input"] = {
            "dynamic": [1, 3, -1, 6],
            "channels": [1, 4, 4, 6],
            "batch": [2, 3, 4, 6],
        }[mismatch]
    with pytest.raises(ValueError):
        runtime.engine_metadata(engine)


def test_build_engine_passes_static_fp32_contract_to_native_builder(native, monkeypatch) -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return native.engine, native.context

    native.handler.tensorRTEngineCreator = create
    monkeypatch.setattr(runtime, "load_upstream", lambda gpu_id: calls.append(gpu_id))
    monkeypatch.setattr(runtime, "runtime_identity", lambda: {"tensorrt": "11", "torch": "test"})
    monkeypatch.setattr(runtime, "adapter_sha256", lambda: "a" * 64)
    sidecar = runtime.build_engine(native.onnx, native.path, 6, 4, 2)
    assert calls == [
        2,
        {
            "modelPath": str(native.onnx),
            "enginePath": str(native.path),
            "fp16": False,
            "inputsMin": [1, 3, 4, 6],
            "inputsOpt": [1, 3, 4, 6],
            "inputsMax": [1, 3, 4, 6],
            "forceStatic": True,
        },
    ]
    assert sidecar["io_precision"] == "fp32"
    assert sidecar["input_profile"] is None
    assert sidecar["builder"] == "tas-native"
    assert sidecar["engine_sha256"] == sha256_file(native.path)
    assert sidecar["model_sha256"] == sha256_file(native.onnx)


@pytest.mark.parametrize(
    "change",
    [
        {"preset": "p1"},
        {"b_frames": 2},
        {"spatial_aq": 0},
        {"gop_frames": 24.0},
        {"codec": "av1"},
        {"extra": "ignored?"},
    ],
)
def test_encoder_contract_is_exact_not_coercible(change) -> None:
    contract = NvencCbrContract(35_000_000, 24)
    assert runtime.encoder_contract(contract.as_dict()) == contract
    with pytest.raises((ValueError, KeyError)):
        runtime.encoder_contract({**contract.as_dict(), **change})


def test_configured_encoders_are_equivalent_and_restored(monkeypatch) -> None:
    settings = SimpleNamespace(
        matchEncoder=lambda _method: [], matchNeluxEncoder=lambda _method: {}
    )
    buffers = SimpleNamespace(**vars(settings))
    monkeypatch.setitem(sys.modules, "src.io.encodingSettings", settings)
    monkeypatch.setitem(sys.modules, "src.io.ffmpegSettings", buffers)
    originals = [vars(module).copy() for module in (settings, buffers)]
    contract = NvencCbrContract(35_000_000, 24)
    with ExitStack() as patches:
        runtime.configure_encoders(contract, patches=patches)
        assert settings.matchEncoder is buffers.matchEncoder
        assert settings.matchNeluxEncoder is buffers.matchNeluxEncoder
        assert settings.matchEncoder("nvenc_h264") == contract.ffmpeg_options()
        options = settings.matchNeluxEncoder("nvenc_h264_nelux")
        assert options["codec"] == "h264_nvenc"
        assert options["pixel_format"] == "yuv420p"
        assert options["bit_rate"] == 35_000_000
        assert options["preset"] == "p4"
        assert options["cq"] is None
        original_options = dict(
            zip(contract.ffmpeg_options()[::2], contract.ffmpeg_options()[1::2], strict=True)
        )
        for key, value in options["options"].items():
            assert original_options["-b:v" if key == "b" else f"-{key}"] == value
        assert len(options["options"]) == len(original_options) - 2
        with pytest.raises(ValueError):
            settings.matchEncoder("x264")
        with pytest.raises(ValueError):
            settings.matchNeluxEncoder("nvenc_hevc_nelux")
    assert [vars(module) for module in (settings, buffers)] == originals


def test_nelux_options_match_video_encoder_keyword_signature() -> None:
    class VideoEncoder:
        def __init__(
            self,
            output,
            *,
            width,
            height,
            fps,
            codec,
            preset,
            cq,
            bit_rate,
            pixel_format,
            options,
        ):
            self.bit_rate = bit_rate
            self.options = options
            self.configuration = (output, width, height, fps, codec, preset, cq, pixel_format)

    contract = NvencCbrContract(35_000_000, 24)
    kwargs = runtime.nelux_encoder_options(contract)
    encoder = VideoEncoder("output.mp4", width=2560, height=1440, fps=24, **kwargs)

    assert encoder.bit_rate == contract.bitrate_bps
    assert encoder.options["b"] == str(contract.bitrate_bps)
    assert encoder.configuration == (
        "output.mp4",
        2560,
        1440,
        24,
        "h264_nvenc",
        "p4",
        None,
        "yuv420p",
    )

    misspelled_kwargs = {**kwargs, "bitrate": kwargs["bit_rate"]}
    del misspelled_kwargs["bit_rate"]
    with pytest.raises(TypeError, match="unexpected keyword argument 'bitrate'"):
        VideoEncoder("output.mp4", width=2560, height=1440, fps=24, **misspelled_kwargs)


def test_load_upstream_selects_gpu_before_cuda_initialization(monkeypatch, tmp_path) -> None:
    (tmp_path / "main.py").write_text("# fake upstream")
    monkeypatch.setenv("TRTVIDEO_TAS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", runtime.TAS_REVISION)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    calls = []

    def available():
        calls.append(os.environ["CUDA_VISIBLE_DEVICES"])
        return True

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=available,
                set_device=lambda value: calls.append(value),
            )
        ),
    )
    constants = SimpleNamespace()
    main = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "src.constants", constants)
    monkeypatch.setitem(sys.modules, "main", main)
    assert runtime.load_upstream(2) is main
    assert calls == ["2", 0]
    assert constants.SYSTEM == "Linux"
    assert str(tmp_path) == constants.WHEREAMIRUNFROM


def test_load_upstream_rejects_cpu_fallback(monkeypatch, tmp_path) -> None:
    (tmp_path / "main.py").write_text("# fake")
    monkeypatch.setenv("TRTVIDEO_TAS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", runtime.TAS_REVISION)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(RuntimeError, match="CPU fallback"):
        runtime.load_upstream()


def test_load_upstream_checks_pin_before_importing_runtime(monkeypatch, tmp_path) -> None:
    (tmp_path / "main.py").write_text("# fake")
    monkeypatch.setenv("TRTVIDEO_TAS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRTVIDEO_TAS_REVISION", "wrong")
    with pytest.raises(RuntimeError, match="source revision"):
        runtime.load_upstream()
