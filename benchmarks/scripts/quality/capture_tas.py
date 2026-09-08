"""Capture native TAS preprocessing or TensorRT inference from shared RGB."""

from __future__ import annotations

import argparse
import importlib
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.benchmark import implementation_config, load_json
from benchmarks.scripts.contracts.engine import load_engine_contract, validate_tas_engine_contract
from benchmarks.scripts.quality.model_space import (
    CaptureManifest,
    ModelSpaceError,
    TensorArtifact,
    create_tensor_artifact,
    parse_frame_indices,
    write_capture_manifest,
)
from benchmarks.scripts.runners.tas_profile import (
    add_execution_profile_arguments,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runners.tas_runtime import load_upstream, pin_engine
from benchmarks.scripts.runtime.environment import collect_image_identity, sha256_file
from benchmarks.scripts.workloads.manifest import (
    find_clip_variant,
    find_model_variant,
    load_manifest,
    repo_path,
)


def _save_tensor(
    tensor: Any, stage: str, frame_index: int, shape: tuple[int, int, int], output_dir: Path
) -> TensorArtifact:
    array = tensor.detach().float().cpu().contiguous().numpy()
    if tuple(array.shape) not in (shape, (1, *shape)):
        raise ModelSpaceError(
            "TAS capture tensor must match the declared RGB CHW or batch-1 NCHW shape"
        )
    array = array.reshape(shape)
    path = output_dir / f"{stage}.frame-{frame_index:06d}.f32"
    array.astype("<f4", copy=False).tofile(path)
    return create_tensor_artifact(
        stage=stage,
        frame_index=frame_index,
        shape=shape,
        path=path,
        root=output_dir,
    )


def _preprocessing(
    input_path: Path,
    output_dir: Path,
    indices: tuple[int, ...],
    shape: tuple[int, int, int],
    decode_method: str,
) -> list[TensorArtifact]:
    buffers = importlib.import_module("src.io.ffmpegSettings")
    reader = buffers.BuildBuffer(
        videoInput=str(input_path),
        half=False,
        width=shape[2],
        height=shape[1],
        toTorch=True,
        decode_method=decode_method,
    )
    native_decode = reader._decodeWithnelux

    def decode(method: str) -> Any:
        if method != decode_method:
            raise RuntimeError("TAS preprocessing attempted a decoder fallback")
        return native_decode(method)

    def no_opencv() -> Any:
        raise RuntimeError("OpenCV fallback is forbidden in TAS quality capture")

    reader._decodeWithnelux = decode
    reader.decodeWithOpenCV = no_opencv
    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    artifacts: list[TensorArtifact] = []
    error: Exception | None = None
    index = 0
    while True:
        frame = reader.read()
        if frame is None:
            break
        try:
            if error is None and index in indices:
                artifacts.append(_save_tensor(frame, "input", index, shape, output_dir))
        except Exception as exc:
            # Drain the native producer before surfacing a capture failure.
            error = exc
        index += 1
    worker.join()
    if error is not None:
        raise error
    if reader.decodeError is not None or reader.decodeMethod != decode_method:
        raise ModelSpaceError(f"TAS preprocessing failed: {reader.decodeError}")
    if len(artifacts) != len(indices):
        raise ModelSpaceError("TAS preprocessing did not capture every requested frame")
    return artifacts


def capture(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    implementation = implementation_config(load_json(Path(args.implementations)), "tas")
    profile = resolve_execution_profile(args)
    validate_declared_profile(implementation, profile)
    clip = find_clip_variant(manifest, args.variant)
    model = find_model_variant(manifest, args.variant)
    input_path = repo_path(root, clip["path"])
    onnx_path = repo_path(root, model["fp16_path"])
    engine_path = Path(args.engine)
    sidecar, _ = load_engine_contract(engine_path)
    validate_tas_engine_contract(sidecar, manifest, args.variant, onnx_path, implementation)
    indices = parse_frame_indices(
        args.frame_indices
        or ",".join(map(str, manifest["quality"]["model_space"]["frame_indices"])),
        frame_count=int(manifest["clip"]["frames"]),
    )
    input_shape = (3, int(model["input_height"]), int(model["input_width"]))
    output_shape = (
        3,
        int(clip["benchmark_output"]["height"]),
        int(clip["benchmark_output"]["width"]),
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ModelSpaceError(f"TAS capture directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = Path(args.shared_input_manifest) if args.shared_input_manifest else None
    load_upstream(args.gpu_id)
    artifacts: list[TensorArtifact]
    if canonical_path is None:
        artifacts = _preprocessing(
            input_path, output_dir, indices, input_shape, profile.decode_method
        )
    else:
        canonical = CaptureManifest.load(canonical_path)
        if (
            canonical.capture_scope != "production-reference"
            or canonical.workload_id != manifest["id"]
            or canonical.variant != args.variant
            or canonical.input_sha256 != sha256_file(input_path)
            or canonical.onnx_sha256 != sha256_file(onnx_path)
        ):
            raise ModelSpaceError("Shared RGB capture does not match the production reference")
        tensors = {item.frame_index: item for item in canonical.artifacts if item.stage == "input"}
        if set(tensors) != set(indices):
            raise ModelSpaceError("Shared RGB capture has the wrong frame set")
        state: dict[str, Any] = {}
        with ExitStack() as patches:
            pin_engine(engine_path, onnx_path, state, patches=patches)
            native_class = importlib.import_module("src.upscale.tensorrt").UniversalTensorRT
            native = native_class(
                upscaleMethod="shufflecugan-tensorrt",
                upscaleFactor=2,
                half=False,
                width=input_shape[2],
                height=input_shape[1],
                customModel=str(onnx_path),
                forceStatic=True,
            )
            numpy = importlib.import_module("numpy")
            torch = importlib.import_module("torch")
            artifacts = []
            for index in indices:
                item = tensors[index]
                source = canonical_path.parent / item.path
                if item.shape != input_shape or sha256_file(source) != item.sha256:
                    raise ModelSpaceError("Shared RGB tensor shape/hash mismatch")
                values = numpy.fromfile(source, dtype="<f4").reshape((1, *input_shape))
                frame = torch.from_numpy(values).to(device="cuda", dtype=torch.float32)
                output = native(frame, None)
                # Capture the actual binding after TAS's native input copy, not a
                # duplicate of the supplied file masquerading as measured input.
                artifacts.append(
                    _save_tensor(native.dummyInput, "input", index, input_shape, output_dir)
                )
                artifacts.append(_save_tensor(output, "output", index, output_shape, output_dir))
            if not state.get("engine_reused"):
                raise ModelSpaceError("TAS inference capture did not use the prebuilt engine")
    path = output_dir / "manifest.json"
    write_capture_manifest(
        path,
        implementation="TheAnimeScripter",
        capture_scope="shared-input-inference" if canonical_path else "production-preprocessing",
        workload_id=manifest["id"],
        variant=args.variant,
        input_sha256=sha256_file(input_path),
        onnx_sha256=sha256_file(onnx_path),
        engine_sha256=sha256_file(engine_path),
        image=collect_image_identity(default_reference=str(implementation["image"])),
        execution_profile=profile.as_parameters(),
        artifacts=artifacts,
        canonical_input_manifest_sha256=sha256_file(canonical_path) if canonical_path else None,
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["tas"], default="tas")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--implementations", default="/app/benchmarks/implementations.json")
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/app")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--shared-input-manifest")
    parser.add_argument("--frame-indices")
    add_execution_profile_arguments(parser)
    return parser


def main() -> None:
    try:
        path = capture(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Tensor capture written: {path}")


if __name__ == "__main__":
    main()
