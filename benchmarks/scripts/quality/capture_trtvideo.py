#!/usr/bin/env python3
"""Capture trtvideo model input/output tensors before RGB-to-YUV conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    load_engine_contract,
    validate_static_engine_contract,
)
from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    TensorArtifact,
    create_tensor_artifact,
    parse_frame_indices,
    write_capture_manifest,
)
from benchmarks.scripts.runtime.environment import collect_image_identity, sha256_file
from benchmarks.scripts.workloads.manifest import (
    WorkloadError,
    find_clip_variant,
    find_model_variant,
    load_manifest,
    repo_path,
)

_DECODE_BATCH_SIZE = 8


def _quality_frame_indices(
    manifest: dict[str, Any],
    override: str | None,
) -> tuple[int, ...]:
    value = (
        override
        if override is not None
        else ",".join(
            str(index)
            for index in manifest["quality"]["model_space"]["frame_indices"]
        )
    )
    return parse_frame_indices(value, frame_count=int(manifest["clip"]["frames"]))


def _write_gpu_tensor(
    tensor: Any,
    *,
    stage: str,
    frame_index: int,
    output_dir: Path,
) -> TensorArtifact:
    import cuda.bindings.runtime as cudart
    import numpy as np

    path = output_dir / f"{stage}.frame-{frame_index:06d}.f32"
    host = _copy_gpu_tensor_to_host(tensor, cudart=cudart, np=np)
    contiguous = np.ascontiguousarray(host[0], dtype="<f4")
    shape = (
        int(contiguous.shape[0]),
        int(contiguous.shape[1]),
        int(contiguous.shape[2]),
    )
    contiguous.tofile(path)
    return create_tensor_artifact(
        stage=stage,
        frame_index=frame_index,
        shape=shape,
        path=path,
        root=output_dir,
    )


def _copy_gpu_tensor_to_host(tensor: Any, *, cudart: Any, np: Any) -> Any:
    """Copy a possibly pitch-padded NCHW CV-CUDA tensor into host memory."""
    interface = tensor.cuda().__cuda_array_interface__
    shape = tuple(int(value) for value in interface["shape"])
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ModelSpaceError(f"Unexpected captured tensor shape: {shape}")

    dtype = np.dtype(str(interface["typestr"]))
    if dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise ModelSpaceError(f"Unexpected captured tensor dtype: {dtype}")
    strides = interface.get("strides")
    if strides is None:
        n_stride = shape[1] * shape[2] * shape[3] * dtype.itemsize
        c_stride = shape[2] * shape[3] * dtype.itemsize
        h_stride = shape[3] * dtype.itemsize
        w_stride = dtype.itemsize
    else:
        n_stride, c_stride, h_stride, w_stride = (
            int(value) for value in strides
        )
    if w_stride != dtype.itemsize:
        raise ModelSpaceError(
            f"Captured tensor width is not contiguous: strides={strides}"
        )

    device_pointer = int(interface["data"][0])
    host = np.empty(shape, dtype=dtype)
    destination_pointer = int(host.ctypes.data)
    row_bytes = shape[3] * dtype.itemsize
    for batch_index in range(shape[0]):
        for channel_index in range(shape[1]):
            result = cudart.cudaMemcpy2D(
                destination_pointer
                + batch_index * host.strides[0]
                + channel_index * host.strides[1],
                host.strides[2],
                device_pointer
                + batch_index * n_stride
                + channel_index * c_stride,
                h_stride,
                row_bytes,
                shape[2],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            if result[0] != cudart.cudaError_t.cudaSuccess:
                raise ModelSpaceError(
                    f"Failed to copy captured tensor to host: {result[0]}"
                )
    return host


def capture(args: argparse.Namespace) -> Path:
    """Run the production decode/preprocess/inference path for selected frames."""
    import PyNvVideoCodec as nvc

    from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
    from trtvideo.video.frames import iter_limited_frames
    from trtvideo.video.nvcodec.decoder import iter_locked_decode_frames
    from trtvideo.video.nvcodec.processor import NvcodecFrameProcessor
    from trtvideo.video.probe import probe_video

    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    clip_variant = find_clip_variant(manifest, args.variant)
    model_variant = find_model_variant(manifest, args.variant)
    input_path = repo_path(root, clip_variant["path"])
    onnx_path = repo_path(root, model_variant["fp16_path"])
    engine_path = Path(args.engine)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = _quality_frame_indices(manifest, args.frame_indices)

    sidecar, _ = load_engine_contract(engine_path)
    validate_static_engine_contract(
        sidecar,
        manifest,
        args.variant,
        onnx_path,
    )
    info = probe_video(str(input_path))
    expected_dimensions = (
        int(model_variant["input_width"]),
        int(model_variant["input_height"]),
    )
    if (info.width, info.height) != expected_dimensions:
        raise ModelSpaceError(
            "Input dimensions do not match the model-space contract: "
            f"{info.width}x{info.height} != "
            f"{expected_dimensions[0]}x{expected_dimensions[1]}"
        )
    if info.pix_fmt not in {"nv12", "yuv420p"}:
        raise ModelSpaceError(f"Unsupported model-space input format: {info.pix_fmt}")
    if info.color_space != "bt709" or info.color_range != "tv":
        raise ModelSpaceError(
            "Model-space capture requires limited-range BT.709 input"
        )

    runtime = CvcudaTensorRTRuntime(
        str(engine_path),
        quiet=True,
        gpu_id=args.gpu_id,
    )
    frame_processor = NvcodecFrameProcessor(
        runtime,
        color_spec_name="bt709",
    )
    decoder = nvc.ThreadedDecoder(
        enc_file_path=str(input_path),
        buffer_size=_DECODE_BATCH_SIZE,
        gpu_id=args.gpu_id,
        cuda_context=0,
        cuda_stream=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
    )

    selected = set(frame_indices)
    artifacts: list[TensorArtifact] = []
    decoded_frames = iter_locked_decode_frames(
        decoder.get_batch_frames,
        batch_size=_DECODE_BATCH_SIZE,
        release_batch=runtime.synchronize,
    )
    for frame_index, raw_frame in enumerate(
        iter_limited_frames(
            decoded_frames,
            limit=frame_indices[-1] + 1,
        )
    ):
        if frame_index not in selected:
            continue
        nv12_tensor = frame_processor.wrap_nv12(raw_frame)
        frame_processor.preprocess(nv12_tensor)
        runtime.synchronize()
        artifacts.append(
            _write_gpu_tensor(
                runtime.gpu_input,
                stage="input",
                frame_index=frame_index,
                output_dir=output_dir,
            )
        )
        model_output = frame_processor.infer()
        runtime.synchronize()
        artifacts.append(
            _write_gpu_tensor(
                model_output,
                stage="output",
                frame_index=frame_index,
                output_dir=output_dir,
            )
        )

    captured_indices = sorted({artifact.frame_index for artifact in artifacts})
    if captured_indices != list(frame_indices):
        raise ModelSpaceError(
            f"Decoder produced only model-space frames {captured_indices}; "
            f"expected {list(frame_indices)}"
        )

    manifest_path = output_dir / "manifest.json"
    write_capture_manifest(
        manifest_path,
        implementation="trtvideo",
        workload_id=manifest["id"],
        variant=args.variant,
        input_sha256=sha256_file(input_path),
        onnx_sha256=sha256_file(onnx_path),
        engine_sha256=sha256_file(engine_path),
        image=collect_image_identity(),
        execution_profile={
            "execution_profile": args.execution_profile,
            "cuda_graph": False,
        },
        artifacts=artifacts,
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], default="1080p")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/app")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--execution-profile",
        choices=["upstream-default", "tuned"],
        default="upstream-default",
    )
    parser.add_argument(
        "--frame-indices",
        default=None,
        help="Override canonical comma-separated zero-based frame indices",
    )
    return parser


def main() -> None:
    try:
        manifest_path = capture(build_parser().parse_args())
    except (
        EngineContractError,
        ModelSpaceError,
        OSError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Model-space capture written: {manifest_path}")


if __name__ == "__main__":
    main()
