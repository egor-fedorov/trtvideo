"""GPU-only video upscaling pipeline via PyNvVideoCodec + TensorRT."""

import argparse
import os
import subprocess
import tempfile
import types
from dataclasses import dataclass

import cvcuda
import PyNvVideoCodec as nvc
import torch

from upscaler.pipeline import BasePipeline

# ---------------------------------------------------------------------------
# NV12 <-> RGB color conversion (GPU, cvcuda)
# ---------------------------------------------------------------------------


def nv12_to_rgb_into(nv12, height, width, rgb_buf, nv12_buf=None):
    """NV12 -> RGB on GPU via cvcuda.

    Args:
        nv12: torch.Tensor (H*3//2, W) or (H*3//2, pitch) uint8 on GPU.
        height: Frame height.
        width: Frame width.
        rgb_buf: Preallocated torch.Tensor (H, W, 3) uint8 on GPU.
        nv12_buf: Optional preallocated contiguous torch.Tensor (H*3//2, W) uint8 on GPU.

    Returns:
        torch.Tensor (H, W, 3) uint8 on GPU.
    """
    if nv12.ndim == 2 and nv12.shape[1] != width:
        nv12 = nv12[:, :width]
    if not nv12.is_contiguous():
        if nv12_buf is None:
            nv12 = nv12.contiguous()
        else:
            nv12_buf.copy_(nv12)
            nv12 = nv12_buf
    nv12_nhwc = nv12.reshape(1, height * 3 // 2, width, 1)
    nv12_cv = cvcuda.as_tensor(nv12_nhwc, "NHWC")
    rgb_cv = cvcuda.as_tensor(rgb_buf.reshape(1, height, width, 3), "NHWC")
    cvcuda.cvtcolor_into(rgb_cv, nv12_cv, cvcuda.ColorConversion.YUV2RGB_NV12)
    return rgb_buf


def nv12_to_rgb(nv12, height, width):
    """NV12 -> RGB on GPU via cvcuda with a newly allocated output buffer."""
    rgb_buf = torch.empty(height, width, 3, dtype=torch.uint8, device="cuda")
    return nv12_to_rgb_into(nv12, height, width, rgb_buf)


def rgb_to_nv12_into(rgb, nv12_buf):
    """RGB -> NV12 on GPU via cvcuda.

    Args:
        rgb: torch.Tensor (H, W, 3) uint8 on GPU.
        nv12_buf: Preallocated torch.Tensor (H*3//2, W) uint8 on GPU.

    Returns:
        torch.Tensor (H*3//2, W) uint8 on GPU.
    """
    h, w = rgb.shape[:2]
    rgb_nhwc = rgb.unsqueeze(0)
    rgb_cv = cvcuda.as_tensor(rgb_nhwc, "NHWC")
    nv12_cv = cvcuda.as_tensor(nv12_buf.reshape(1, h * 3 // 2, w, 1), "NHWC")
    cvcuda.cvtcolor_into(nv12_cv, rgb_cv, cvcuda.ColorConversion.RGB2YUV_NV12)
    return nv12_buf


def rgb_to_nv12(rgb):
    """RGB -> NV12 on GPU via cvcuda with a newly allocated output buffer."""
    h, w = rgb.shape[:2]
    nv12_buf = torch.empty(h * 3 // 2, w, dtype=torch.uint8, device=rgb.device)
    return rgb_to_nv12_into(rgb, nv12_buf)


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


def crf_to_bitrate(crf, width, height, fps):
    """Estimate bitrate from a CRF-like value.

    bpp ~ 0.1 at CRF 23, doubles every 6 units.
    """
    bpp = 0.1 * (2.0 ** ((23 - crf) / 6.0))
    return int(bpp * width * height * fps)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GpuPipeline(BasePipeline):
    """Pipeline: NVDEC -> NV12 -> RGB (cvcuda) -> TRT -> RGB -> NV12 (cvcuda) -> NVENC."""

    DESCRIPTION = "TensorRT Video Upscaler (NVDEC/NVENC backend)"
    BACKEND_NAME = "nvcodec"
    _DECODE_BATCH_SIZE = 8

    def __init__(self):
        self._decoder = None
        self._encoder = None
        self._tmp_raw_path: str = ""
        self._raw_file = None
        self._buffer_pool: FrameBufferPool | None = None
        super().__init__()

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--crf", type=int, default=18, help="Quality (mapped to bitrate for NVENC)"
        )
        parser.add_argument("--codec", default="h264", choices=["h264", "hevc"], help="NVENC codec")

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

    def setup_decoder(self) -> None:
        self.log("Initializing NVDEC...")
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
        bitrate = crf_to_bitrate(
            self.args.crf,
            runtime.output_w,
            runtime.output_h,
            self.info["fps"],
        )
        self.log(f"Initializing NVENC ({self.args.codec}, {bitrate / 1e6:.1f} Mbps)...")

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
            fps=int(round(self.info["fps"])),
        )
        self._raw_file = open(self._tmp_raw_path, "wb")

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
            rgb = nv12_to_rgb_into(nv12_tensor, in_h, in_w, pool.rgb_in, pool.nv12_in)
            upscaled = self._infer_gpu(rgb)
            nv12_out = rgb_to_nv12_into(upscaled, pool.nv12_out)
            self._write_bitstream(self._encoder.Encode(_patch_dlpack(nv12_out)))

    def _infer_gpu(self, rgb_hwc):
        """TRT inference entirely on GPU."""
        pool = self._require_buffer_pool()
        return self.require_runtime().infer_rgb_tensor_into(
            rgb_hwc,
            pool.rgb_out,
            input_nchw=pool.nchw_in,
            output_rgb_float=pool.rgb_out_float,
        )

    def _process_frame_profiled(self, nv12_tensor, in_h, in_w):
        """Inference with CUDA event profiling and explicit TRT stream handoff."""
        e0, e1, e2, e3, e4 = (torch.cuda.Event(enable_timing=True) for _ in range(5))
        cur_stream = torch.cuda.current_stream()
        runtime = self.require_runtime()
        trt_stream = runtime.stream

        e0.record(cur_stream)
        pool = self._require_buffer_pool()
        rgb = nv12_to_rgb_into(nv12_tensor, in_h, in_w, pool.rgb_in, pool.nv12_in)
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
        nv12_out = rgb_to_nv12_into(upscaled, pool.nv12_out)
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
