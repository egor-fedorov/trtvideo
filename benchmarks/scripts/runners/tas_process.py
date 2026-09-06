"""Run the pinned TAS CLI with benchmark configuration and boundary observers."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks.scripts.runners.tas_runtime import (
    TAS_REVISION,
    adapter_sha256,
    configure_encoders,
    encoder_contract,
    load_upstream,
    patch_attribute,
    pin_engine,
    runtime_identity,
)
from benchmarks.scripts.runtime.environment import write_json
from trtvideo.benchmarking.lifecycle import FrameLifecycleMarkers, write_frame_markers

TensorObserver = Callable[[int, Any, Any], None]


def native_arguments(args: argparse.Namespace) -> list[str]:
    duration = Fraction(args.frames, 1) / Fraction(args.fps)
    return [
        "tas",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--upscale",
        "--upscale_method",
        "shufflecugan-tensorrt",
        "--upscale_factor",
        "2",
        "--custom_model",
        str(args.onnx),
        "--half",
        "false",
        "--static",
        "--decode_method",
        args.decode_method,
        "--encode_method",
        "nvenc_h264_nelux" if args.writer == "nelux" else "nvenc_h264",
        "--outpoint",
        format(float(duration), ".17g"),
    ]


def _hermetic_dependencies(patches: ExitStack) -> None:
    """Replace interactive installation, not TAS's frame processing machinery."""
    dependencies: Any = importlib.import_module("src.infra.dependencyHandler")

    def check(self: Any) -> bool:
        for name in ("torch", "tensorrt", "nelux", "onnxruntime", "barflow"):
            importlib.metadata.version(name)
        return True

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Downloads/installations are forbidden in TAS benchmark processes")

    patch_attribute(patches, dependencies.DependencyChecker, "ensureDependencies", check)
    patch_attribute(patches, dependencies, "installDependencies", forbidden)
    patch_attribute(patches, importlib.import_module("src.infra.getFFMPEG"), "getFFMPEG", forbidden)


def run(args: argparse.Namespace, *, on_tensor: TensorObserver | None = None) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "invalid",
        "errors": [],
        "source_revision": TAS_REVISION,
        "adapter_sha256": None,
        "engine_sha256": None,
        "onnx_sha256": None,
        "engine_reused": False,
        "cuda_graph": True,
        "processed_frames": 0,
        "decode_method": None,
        "writer": None,
    }
    try:
        evidence["adapter_sha256"] = adapter_sha256()
        if args.frames <= 0 or Fraction(args.fps) <= 0:
            raise ValueError("TAS requires a positive frame count and FPS")
        contract = encoder_contract(json.loads(args.encoder_contract.read_text()))
        # The parent hashes read-only assets before launch. Avoid a second full
        # engine read in measured startup solely for provenance.
        sidecar = json.loads(Path(f"{args.engine}.json").read_text())
        evidence.update(engine_sha256=sidecar["engine_sha256"], onnx_sha256=sidecar["model_sha256"])
        with ExitStack() as patches:
            _run_upstream(args, contract, evidence, on_tensor, patches)
        evidence["status"] = "valid"
    except BaseException as exc:
        evidence["errors"].append(str(exc) or type(exc).__name__)
        raise
    finally:
        if args.runtime_evidence is not None:
            write_json(args.runtime_evidence, evidence)
    return evidence


def _run_upstream(
    args: argparse.Namespace,
    contract: Any,
    evidence: dict[str, Any],
    on_tensor: TensorObserver | None,
    patches: ExitStack,
) -> None:
    upstream = load_upstream(args.gpu_id)
    pin_engine(args.engine, args.onnx, evidence, patches=patches)
    configure_encoders(contract, patches=patches)
    _hermetic_dependencies(patches)
    buffers: Any = importlib.import_module("src.io.ffmpegSettings")
    observed: dict[str, Any] = {"decoders": [], "readers": [], "writers": []}
    original_decode = buffers.BuildBuffer._decodeWithnelux
    original_writer = buffers.createWriteBuffer
    original_write = upstream.VideoProcessor._writeOut
    times: list[int] = []

    def decode(reader: Any, method: str) -> Any:
        observed["decoders"].append(method)
        observed["readers"].append(reader)
        if method != args.decode_method:
            raise RuntimeError("TAS decoder fallback is forbidden")
        return original_decode(reader, method)

    def no_opencv(reader: Any) -> Any:
        observed["decoders"].append("opencv")
        raise RuntimeError("OpenCV fallback is forbidden in TAS benchmarks")

    def writer(*pos: Any, **kw: Any) -> Any:
        instance = original_writer(*pos, **kw)
        observed["writers"].append(instance)
        return instance

    def completed(processor: Any, frame: Any) -> None:
        original_write(processor, frame)
        timestamp = time.perf_counter_ns()
        if not times:
            times.append(timestamp)
        if len(times) == 1:
            times.append(timestamp)
        times[1] = timestamp
        evidence["processed_frames"] += 1

    patch_attribute(patches, buffers.BuildBuffer, "_decodeWithnelux", decode)
    patch_attribute(patches, buffers.BuildBuffer, "decodeWithOpenCV", no_opencv)
    patch_attribute(patches, buffers, "createWriteBuffer", writer)
    patch_attribute(patches, upstream.VideoProcessor, "_writeOut", completed)
    if on_tensor is not None:
        native_class = importlib.import_module("src.upscale.tensorrt").UniversalTensorRT
        original_inference = native_class.__call__
        capture_index = 0

        def capture(model: Any, frame: Any, next_frame: Any = None) -> Any:
            nonlocal capture_index
            output = original_inference(model, frame, next_frame)
            on_tensor(capture_index, model.dummyInput, output)
            capture_index += 1
            return output

        patch_attribute(patches, native_class, "__call__", capture)

    patch_attribute(patches, sys, "argv", native_arguments(args))
    try:
        upstream.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(f"TAS exited with code {exc.code}") from exc
    decoders = observed["decoders"]
    if decoders != [args.decode_method]:
        raise RuntimeError(f"TAS decoder fallback or unexpected decode invocation: {decoders}")
    reader = observed["readers"][0]
    evidence["decode_method"] = reader.decodeMethod
    if reader.decodeMethod != args.decode_method or reader.decodeError is not None:
        raise RuntimeError(f"TAS decoder failed or changed backend: {reader.decodeError}")
    writers = observed["writers"]
    if len(writers) != 1:
        raise RuntimeError("TAS did not create exactly one output writer")
    if isinstance(writers[0], buffers.NeluxWriteBuffer):
        actual = "nelux"
    elif isinstance(writers[0], buffers.WriteBuffer):
        actual = "ffmpeg"
    else:
        raise RuntimeError("TAS created an unsupported output writer")
    evidence["writer"] = actual
    if actual != args.writer:
        raise RuntimeError(f"TAS writer fallback: requested {args.writer}, got {actual}")
    if getattr(writers[0], "encodeError", None) is not None:
        raise RuntimeError(f"TAS encoder failed: {writers[0].encodeError}")
    frame_count = evidence["processed_frames"]
    if frame_count != args.frames or not evidence["engine_reused"]:
        raise RuntimeError(
            f"TAS processed {frame_count}/{args.frames} frames or did not load engine"
        )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("TAS produced no output video")
    evidence["runtime"] = runtime_identity()
    if args.lifecycle_output is not None:
        write_frame_markers(
            args.lifecycle_output,
            FrameLifecycleMarkers(
                first_frame_completed_ns=times[0],
                last_frame_completed_ns=times[1],
                processed_frames=frame_count,
                instrumentation="tas-output-queue-submission",
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "onnx", "input", "output", "encoder-contract"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", default="24/1")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--decode-method", choices=("cpu", "nvdec"), required=True)
    parser.add_argument("--writer", choices=("ffmpeg", "nelux"), required=True)
    parser.add_argument("--lifecycle-output", type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
