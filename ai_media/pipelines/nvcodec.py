"""GPU-only video upscaling pipeline via PyNvVideoCodec + TensorRT."""

import argparse
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from typing import Any

import PyNvVideoCodec as nvc
import torch

from ai_media.pipelines.base import BasePipeline
from ai_media.video.bitrate import auto_bitrate_from_source
from ai_media.video.colorspace import nv12_to_rgb_into, rgb_to_nv12_into
from ai_media.video.fps import format_nvenc_fps, gop_size_for_one_second


@dataclass
class FrameBufferPool:
    """Per-job GPU buffers reused by the NVDEC/NVENC hot path."""

    nv12_in: torch.Tensor
    rgb_in: torch.Tensor
    nchw_in: torch.Tensor
    rgb_out: torch.Tensor
    rgb_out_float: torch.Tensor
    nv12_out: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        input_w: int,
        input_h: int,
        output_w: int,
        output_h: int,
        device: torch.device,
        input_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> "FrameBufferPool":
        return cls(
            nv12_in=torch.empty(input_h * 3 // 2, input_w, dtype=torch.uint8, device=device),
            rgb_in=torch.empty(input_h, input_w, 3, dtype=torch.uint8, device=device),
            nchw_in=torch.empty(1, 3, input_h, input_w, dtype=input_dtype, device=device),
            rgb_out=torch.empty(output_h, output_w, 3, dtype=torch.uint8, device=device),
            rgb_out_float=torch.empty(output_h, output_w, 3, dtype=output_dtype, device=device),
            nv12_out=torch.empty(output_h * 3 // 2, output_w, dtype=torch.uint8, device=device),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_dlpack(tensor):
    """Patch __dlpack__ for PyTorch >= 2.x compatibility with PyNvVideoCodec."""

    def _dlpack(self, *args, **kwargs):
        return torch.utils.dlpack.to_dlpack(self)

    tensor.__dlpack__ = types.MethodType(_dlpack, tensor)
    return tensor


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NvcodecPipeline(BasePipeline):
    """Pipeline: NVDEC -> NV12 -> RGB (cvcuda) -> TRT -> RGB -> NV12 (cvcuda) -> NVENC."""

    DESCRIPTION = "TensorRT Video Upscaler (NVDEC/NVENC backend)"
    BACKEND_NAME = "nvcodec"
    _DECODE_BATCH_SIZE = 8

    def __init__(self, args: argparse.Namespace):
        self._decoder: Any | None = None
        self._encoder: Any | None = None
        self._tmp_raw_path: str = ""
        self._raw_file: Any | None = None
        self._buffer_pool: FrameBufferPool | None = None
        self._color_spec_name = "bt709"
        super().__init__(args)

    _GPU_STAGES = [
        "NV12\u2192RGB (cvcuda)",
        "TRT inference",
        "RGB\u2192NV12 (cvcuda)",
        "NVENC encode",
    ]
    _PROFILE_STAGE_KEYS = {
        "NV12\u2192RGB (cvcuda)": "nv12_to_rgb",
        "TRT inference": "trt",
        "RGB\u2192NV12 (cvcuda)": "rgb_to_nv12",
        "NVENC encode": "encode",
    }

    def profile_stage_names(self) -> list[str]:
        return self._GPU_STAGES

    def gpu_stage_names(self) -> list[str]:
        return self._GPU_STAGES

    def profile_stage_key_map(self) -> dict[str, str]:
        return self._PROFILE_STAGE_KEYS

    def _require_buffer_pool(self) -> FrameBufferPool:
        if self._buffer_pool is None:
            raise RuntimeError("Frame buffer pool is not initialized")
        return self._buffer_pool

    def validate_video_input(self, info) -> None:
        super().validate_video_input(info)
        if info.pix_fmt not in {"nv12", "yuv420p"}:
            print(
                "ERROR: nvcodec backend currently supports only 8-bit SDR NV12/yuv420p "
                f"input, got pix_fmt={info.pix_fmt or 'unknown'}. "
                "Use --backend ffmpeg or transcode/tonemap the input to SDR yuv420p first."
            )
            sys.exit(1)

    def setup_decoder(self) -> None:
        self.log("Initializing NVDEC...")
        self._color_spec_name = self.cvcuda_color_spec_name()
        self.log_verbose(f"CV-CUDA color spec: {self._color_spec_name}")
        self._decoder = nvc.ThreadedDecoder(
            enc_file_path=self.args.input,
            buffer_size=self._DECODE_BATCH_SIZE,
            gpu_id=self.args.gpu_id,
            cuda_context=0,
            cuda_stream=0,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
        )

    def setup_encoder(self) -> None:
        runtime = self.require_runtime()
        self._buffer_pool = FrameBufferPool.create(
            input_w=runtime.input_w,
            input_h=runtime.input_h,
            output_w=runtime.output_w,
            output_h=runtime.output_h,
            device=torch.device(f"cuda:{self.args.gpu_id}"),
            input_dtype=runtime.input_dtype,
            output_dtype=runtime.output_dtype,
        )
        bitrate = self._resolve_bitrate(runtime)
        try:
            encoder_fps = format_nvenc_fps(self.info.fps_str)
            gop_size = gop_size_for_one_second(self.info.fps_str)
        except ValueError as exc:
            print(f"ERROR: Unsupported input FPS for NVENC: {exc}")
            sys.exit(1)
        self.log(
            f"Initializing NVENC ({self.args.codec}, {bitrate / 1e6:.1f} Mbps, "
            f"{self._color_spec_name}, fps={encoder_fps}, gop={gop_size}, bframes=0)..."
        )

        raw_ext = ".h264" if self.args.codec == "h264" else ".hevc"
        tmp_fd, self._tmp_raw_path = tempfile.mkstemp(suffix=raw_ext)
        os.close(tmp_fd)

        self._encoder = nvc.CreateEncoder(
            runtime.output_w,
            runtime.output_h,
            "NV12",
            False,
            gpu_id=self.args.gpu_id,
            codec=self.args.codec,
            bitrate=bitrate,
            preset="P4",
            tuning_info="high_quality",
            fps=encoder_fps,
            gop=gop_size,
            idrperiod=gop_size,
            bf=0,
            repeatspspps=1,
        )
        self._raw_file = open(self._tmp_raw_path, "wb")

    def _resolve_bitrate(self, runtime) -> int:
        if self.args.bitrate_mbps is not None:
            if self.args.bitrate_mbps <= 0:
                print("ERROR: --bitrate-mbps must be greater than zero")
                sys.exit(1)
            bitrate = int(self.args.bitrate_mbps * 1_000_000)
            self.log(f"NVENC bitrate: manual {bitrate / 1e6:.1f} Mbps")
            return bitrate

        source_bitrate = self.info.video_bit_rate or self.info.container_bit_rate
        if source_bitrate:
            bitrate, pixel_ratio, fps_ratio = auto_bitrate_from_source(
                source_bitrate=source_bitrate,
                input_w=runtime.input_w,
                input_h=runtime.input_h,
                output_w=runtime.output_w,
                output_h=runtime.output_h,
                input_fps=self.info.fps,
                output_fps=self.info.fps,
            )
            source = "video stream" if self.info.video_bit_rate else "container"
            self.log(
                "NVENC bitrate auto: "
                f"source={source_bitrate / 1e6:.1f} Mbps ({source}), "
                f"pixel_ratio={pixel_ratio:.2f}, "
                f"fps_ratio={fps_ratio:.2f}, "
                f"target={bitrate / 1e6:.1f} Mbps"
            )
            return bitrate

        print(
            "ERROR: nvcodec auto bitrate requires source video bitrate metadata. "
            "Pass --bitrate-mbps explicitly."
        )
        sys.exit(1)

    def decode_frames(self):
        """Yield NV12 frames from NVDEC decoder."""
        while True:
            frames = self._decoder.get_batch_frames(self._DECODE_BATCH_SIZE)
            if not frames:
                break
            yield from frames

    def _write_bitstream(self, bs):
        if bs and self._raw_file:
            self._raw_file.write(bytearray(bs))

    def process_frame(self, raw_frame) -> None:
        nv12_tensor = torch.from_dlpack(raw_frame)
        runtime = self.require_runtime()
        in_h, in_w = runtime.input_h, runtime.input_w

        if self.profiler:
            self._process_frame_profiled(nv12_tensor, in_h, in_w)
        else:
            pool = self._require_buffer_pool()
            stream = runtime.stream
            with torch.cuda.stream(stream):
                rgb = nv12_to_rgb_into(
                    nv12_tensor,
                    in_h,
                    in_w,
                    pool.rgb_in,
                    pool.nv12_in,
                    color_spec=self._color_spec_name,
                )
                upscaled = runtime.infer_rgb_tensor_into(
                    rgb,
                    pool.rgb_out,
                    input_nchw=pool.nchw_in,
                    output_rgb_float=pool.rgb_out_float,
                    stream=stream,
                    synchronize=False,
                )
                nv12_out = rgb_to_nv12_into(
                    upscaled,
                    pool.nv12_out,
                    color_spec=self._color_spec_name,
                )
            stream.synchronize()
            self._write_bitstream(self._encoder.Encode(_patch_dlpack(nv12_out)))

    def _process_frame_profiled(self, nv12_tensor, in_h, in_w):
        """Inference with CUDA event profiling and explicit TRT stream handoff."""
        e0, e1, e2, e3, e4 = (torch.cuda.Event(enable_timing=True) for _ in range(5))
        cur_stream = torch.cuda.current_stream()
        runtime = self.require_runtime()
        trt_stream = runtime.stream

        e0.record(cur_stream)
        pool = self._require_buffer_pool()
        rgb = nv12_to_rgb_into(
            nv12_tensor,
            in_h,
            in_w,
            pool.rgb_in,
            pool.nv12_in,
            color_spec=self._color_spec_name,
        )
        e1.record(cur_stream)

        trt_stream.wait_event(e1)
        upscaled = runtime.infer_rgb_tensor_into(
            rgb,
            pool.rgb_out,
            input_nchw=pool.nchw_in,
            output_rgb_float=pool.rgb_out_float,
            stream=trt_stream,
            synchronize=False,
        )
        e2.record(trt_stream)

        cur_stream.wait_event(e2)
        nv12_out = rgb_to_nv12_into(
            upscaled,
            pool.nv12_out,
            color_spec=self._color_spec_name,
        )
        e3.record(cur_stream)

        self._write_bitstream(self._encoder.Encode(_patch_dlpack(nv12_out)))
        e4.record(cur_stream)

        torch.cuda.synchronize()
        self.profiler.commit((e0, e1, e2, e3, e4))

    def finalize(self) -> None:
        """Flush NVENC and mux raw bitstream to MP4."""
        self._write_bitstream(self._encoder.EndEncode())
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None

        self.log("\nMuxing to MP4...")
        mux_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-r",
            self.info["fps_str"],
            "-i",
            self._tmp_raw_path,
            "-i",
            self.args.input,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-movflags",
            "+faststart",
            *self.ffmpeg_color_metadata_args(),
            *self.ffmpeg_limited_duration_args(),
            self.args.output,
        ]
        self.log_verbose(f"Mux cmd: {' '.join(mux_cmd)}")
        result = subprocess.run(mux_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: ffmpeg mux failed: {result.stderr}")

    def cleanup(self) -> None:
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None
        if self._tmp_raw_path and os.path.exists(self._tmp_raw_path):
            os.unlink(self._tmp_raw_path)
