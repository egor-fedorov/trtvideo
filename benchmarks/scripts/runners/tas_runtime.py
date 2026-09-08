"""Narrow configuration/provenance adapter for an unmodified TAS checkout.

Heavy dependencies are loaded only inside the TAS container. This module never
reimplements decoding, tensor normalization, inference, or encoding.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmarks.scripts.runtime.environment import sha256_file
from trtvideo.video.nvcodec.encoder import NvencCbrContract

TAS_REVISION = "ac259ddf13c191230a4c65a4b251c9fb28884104"


def patch_attribute(patches: ExitStack, target: Any, name: str, value: Any) -> None:
    """Install one temporary upstream adapter and restore it when its scope exits."""
    original = getattr(target, name)
    patches.callback(setattr, target, name, original)
    setattr(target, name, value)


def adapter_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("tas_runtime.py", "tas_process.py"):
        digest.update(name.encode("ascii"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def load_upstream(gpu_id: int = 0) -> ModuleType:
    """Select the device before importing TAS's eagerly-created CUDA checkers."""
    if gpu_id < 0:
        raise ValueError("GPU id must be non-negative")
    root = Path(os.environ.get("TRTVIDEO_TAS_ROOT", "/opt/tas"))
    if not (root / "main.py").is_file():
        raise RuntimeError(f"Pinned TAS source not found at {root}")
    if os.environ.get("TRTVIDEO_TAS_REVISION") != TAS_REVISION:
        raise RuntimeError("TAS image source revision does not match the adapter")
    # TAS addresses cuda:0 internally. NVML accounting stays on the physical id
    # in the parent runner; this child maps that physical GPU to logical zero.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(root))
    torch: Any = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("TAS benchmark requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    constants: Any = importlib.import_module("src.constants")
    constants.SYSTEM = "Linux"
    constants.WHEREAMIRUNFROM = str(root)
    return importlib.import_module("main")


def runtime_identity() -> dict[str, str]:
    root = Path(os.environ.get("TRTVIDEO_TAS_ROOT", "/opt/tas"))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "tensorrt", "nelux")}
    versions["python"] = sys.version.split()[0]
    versions["ffmpeg"] = importlib.import_module("src.infra.getFFMPEG").FFMPEG_AV_VERSION_INFO
    versions["ffmpeg_sha256"] = sha256_file(root / "ffmpeg_shared" / "ffmpeg")
    return versions


def engine_metadata(engine: Any) -> dict[str, Any]:
    trt = importlib.import_module("tensorrt")
    if engine.num_io_tensors != 2:
        raise ValueError("TAS benchmark requires one RGB input and one RGB output")
    result: dict[str, Any] = {}
    for index, key in enumerate(("input", "output")):
        name = engine.get_tensor_name(index)
        mode = trt.TensorIOMode.INPUT if key == "input" else trt.TensorIOMode.OUTPUT
        shape = list(engine.get_tensor_shape(name))
        if engine.get_tensor_mode(name) != mode:
            raise ValueError("TAS requires engine input/output in that binding order")
        if engine.get_tensor_dtype(name) != trt.float32:
            raise ValueError("TAS benchmark requires FP32 tensor bindings")
        if len(shape) != 4 or shape[:2] != [1, 3] or any(value <= 0 for value in shape):
            raise ValueError("TAS benchmark requires static batch-1 RGB NCHW")
        result[key] = {"name": name, "shape": shape, "dtype": "float32"}
    return result


def build_engine(onnx: Path, engine: Path, width: int, height: int, gpu_id: int) -> dict[str, Any]:
    load_upstream(gpu_id)
    handler = importlib.import_module("src.model.trtHandler")
    shape = [1, 3, height, width]
    native, context = handler.tensorRTEngineCreator(
        modelPath=str(onnx),
        enginePath=str(engine),
        fp16=False,
        inputsMin=shape,
        inputsOpt=shape,
        inputsMax=shape,
        forceStatic=True,
    )
    if native is None or context is None or not engine.is_file():
        raise RuntimeError("Native TAS engine build failed")
    metadata = engine_metadata(native)
    versions = runtime_identity()
    return {
        "schema_version": 1,
        **metadata,
        "engine_sha256": sha256_file(engine),
        "model_sha256": sha256_file(onnx),
        "io_precision": "fp32",
        "input_profile": None,
        "tensorrt_version": versions["tensorrt"],
        "builder": "tas-native",
        "builder_flags": ["stronglyTyped"],
        "tas_revision": TAS_REVISION,
        "adapter_sha256": adapter_sha256(),
        "runtime": versions,
        "builder_base_image": os.environ.get("TRTVIDEO_BASE_IMAGE", "unknown"),
    }


def pin_engine(engine: Path, onnx: Path, evidence: dict[str, Any], *, patches: ExitStack) -> None:
    """Keep TAS's loader and inference untouched; prohibit its build fallback."""
    handler: Any = importlib.import_module("src.model.trtHandler")
    native_loader = handler.tensorRTEngineLoader

    def resolve(*args: Any, **kwargs: Any) -> str:
        model_path = kwargs.get("modelPath", args[0] if args else None)
        if model_path is None or Path(model_path).resolve() != onnx.resolve():
            raise RuntimeError("TAS requested an unexpected ONNX")
        return str(engine)

    def load(path: str, *args: Any, **kwargs: Any) -> Any:
        if Path(path).resolve() != engine.resolve() or not engine.is_file():
            raise RuntimeError("TAS requested an unexpected or missing engine")
        native, context = native_loader(path, *args, **kwargs)
        if native is None or context is None:
            raise RuntimeError("TAS could not reuse the prebuilt engine")
        engine_metadata(native)
        evidence["engine_reused"] = True
        return native, context

    def forbid_build(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Engine building is forbidden during a TAS measurement")

    patch_attribute(patches, handler, "tensorRTEngineNameHandler", resolve)
    patch_attribute(patches, handler, "tensorRTEngineLoader", load)
    patch_attribute(patches, handler, "tensorRTEngineCreator", forbid_build)
    shape = json.loads(Path(f"{engine}.json").read_text())["input"]["shape"]
    native_class = importlib.import_module("src.upscale.tensorrt").UniversalTensorRT

    def static_multiple(model: Any) -> int:
        # TAS's square-probe heuristic cannot discover divisibility from a
        # static ONNX. The pinned engine already specifies the exact full frame.
        if [1, 3, model.height, model.width] != shape:
            raise RuntimeError("TAS video shape differs from the declared static engine")
        if Path(model.modelPath).resolve() != onnx.resolve():
            raise RuntimeError("TAS static shape belongs to a different model")
        return 1

    patch_attribute(patches, native_class, "_detectRequiredMultiple", static_multiple)


def encoder_contract(values: dict[str, Any]) -> NvencCbrContract:
    contract = NvencCbrContract(
        bitrate_bps=int(values["target_bitrate_bps"]),
        gop_frames=int(values["gop_frames"]),
        codec=str(values["codec"]),
    )
    expected = contract.as_dict()
    if values != expected or any(
        type(values[key]) is not type(value) for key, value in expected.items()
    ):
        raise ValueError("TAS encoder settings differ from the canonical CBR contract")
    return contract


def nelux_encoder_options(contract: NvencCbrContract) -> dict[str, Any]:
    args = contract.ffmpeg_options()
    options = {args[index].removeprefix("-"): args[index + 1] for index in range(0, len(args), 2)}
    codec = options.pop("c:v")
    options["b"] = options.pop("b:v")
    options.pop("preset")
    return {
        "codec": codec,
        "preset": "p4",
        "cq": None,
        "bit_rate": contract.bitrate_bps,
        "pixel_format": "yuv420p",
        "options": options,
    }


def configure_encoders(contract: NvencCbrContract, *, patches: ExitStack) -> None:
    settings: Any = importlib.import_module("src.io.encodingSettings")
    buffers: Any = importlib.import_module("src.io.ffmpegSettings")

    def ffmpeg(method: str) -> list[str]:
        if method != "nvenc_h264":
            raise ValueError(f"Unexpected TAS FFmpeg encoder: {method}")
        return contract.ffmpeg_options()

    def nelux(method: str) -> dict[str, Any]:
        if method != "nvenc_h264_nelux":
            raise ValueError(f"Unexpected TAS neLux encoder: {method}")
        return nelux_encoder_options(contract)

    for module in (settings, buffers):
        patch_attribute(patches, module, "matchEncoder", ffmpeg)
        patch_attribute(patches, module, "matchNeluxEncoder", nelux)
