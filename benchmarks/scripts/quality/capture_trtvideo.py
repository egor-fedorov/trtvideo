#!/usr/bin/env python3
"""Capture trtvideo model input/output tensors before RGB-to-YUV conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.quality.model_space import (
    ModelSpaceError,
    TensorArtifact,
    create_tensor_artifact,
    parse_frame_indices,
    write_capture_manifest,
)
from benchmarks.scripts.runners.common import (
    CompetitorError,
    find_model_variant,
    find_variant,
    validate_static_engine_contract,
)
from benchmarks.scripts.runtime.environment import collect_image_identity, sha256_file
from benchmarks.scripts.runtime.runner import BenchmarkError, load_engine_contract
from benchmarks.scripts.workloads.manifest import load_manifest, repo_path

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
    import numpy as np

    path = output_dir / f"{stage}.frame-{frame_index:06d}.f32"
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ModelSpaceError(f"Unexpected captured {stage} tensor shape: {array.shape}")
    contiguous = np.ascontiguousarray(array, dtype="<f4")
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


def capture(args: argparse.Namespace) -> Path:
    """Run the production decode/preprocess/inference path for selected frames."""
    import cvcuda
    import PyNvVideoCodec as nvc
    import torch

    from trtvideo.runtime.tensorrt import TensorRTRuntime
    from trtvideo.video.colorspace import nv12_to_rgb_into
    from trtvideo.video.info import get_video_info

    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    clip_variant = find_variant(manifest, args.variant)
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
    info = get_video_info(str(input_path))
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

    runtime = TensorRTRuntime(
        str(engine_path),
        quiet=True,
        gpu_id=args.gpu_id,
        use_cuda_graph=False,
    )
    cvcuda_stream = cvcuda.as_stream(runtime.stream)
    device = torch.device(f"cuda:{args.gpu_id}")
    nv12_buffer = torch.empty(
        runtime.input_h * 3 // 2,
        runtime.input_w,
        dtype=torch.uint8,
        device=device,
    )
    rgb_buffer = torch.empty(
        runtime.input_h,
        runtime.input_w,
        3,
        dtype=torch.uint8,
        device=device,
    )
    model_input = torch.empty(
        runtime.input_shape,
        dtype=runtime.input_dtype,
        device=device,
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
    frame_index = 0
    while frame_index <= frame_indices[-1]:
        frames = decoder.get_batch_frames(_DECODE_BATCH_SIZE)
        if not frames:
            break
        for raw_frame in frames:
            if frame_index in selected:
                with torch.cuda.stream(runtime.stream):
                    nv12_tensor = torch.from_dlpack(raw_frame)
                    rgb = nv12_to_rgb_into(
                        nv12_tensor,
                        runtime.input_h,
                        runtime.input_w,
                        rgb_buffer,
                        nv12_buffer,
                        color_spec="bt709",
                        stream=cvcuda_stream,
                    )
                    model_input.copy_(rgb.permute(2, 0, 1).unsqueeze(0))
                    model_input.div_(255.0)
                    model_output = runtime.infer(
                        {runtime.input_name: model_input},
                        stream=runtime.stream,
                        synchronize=False,
                    )[runtime.output_name]
                runtime.stream.synchronize()
                artifacts.append(
                    _write_gpu_tensor(
                        runtime.gpu_input,
                        stage="input",
                        frame_index=frame_index,
                        output_dir=output_dir,
                    )
                )
                artifacts.append(
                    _write_gpu_tensor(
                        model_output,
                        stage="output",
                        frame_index=frame_index,
                        output_dir=output_dir,
                    )
                )
            frame_index += 1
            if frame_index > frame_indices[-1]:
                break

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
        comparison_class="reference",
        workload_id=manifest["id"],
        variant=args.variant,
        input_sha256=sha256_file(input_path),
        onnx_sha256=sha256_file(onnx_path),
        engine_sha256=sha256_file(engine_path),
        image=collect_image_identity(),
        execution_profile={
            "mode": args.execution_profile,
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
        choices=["parity", "upstream-default", "tuned"],
        default="parity",
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
        BenchmarkError,
        CompetitorError,
        ModelSpaceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Model-space capture written: {manifest_path}")


if __name__ == "__main__":
    main()
