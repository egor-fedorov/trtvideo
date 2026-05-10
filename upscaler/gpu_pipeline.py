"""GPU-only video upscaling pipeline via PyNvVideoCodec + TensorRT."""

import argparse
import os
import subprocess
import tempfile
import types

import cvcuda
import PyNvVideoCodec as nvc
import torch

from upscaler.pipeline import BasePipeline


# ---------------------------------------------------------------------------
# NV12 <-> RGB color conversion (GPU, cvcuda)
# ---------------------------------------------------------------------------


def nv12_to_rgb(nv12, height, width):
    """NV12 -> RGB on GPU via cvcuda.

    Args:
        nv12: torch.Tensor (H*3//2, W) or (H*3//2, pitch) uint8 on GPU.
        height: Frame height.
        width: Frame width.

    Returns:
        torch.Tensor (H, W, 3) uint8 on GPU.
    """
    if nv12.ndim == 2 and nv12.shape[1] != width:
        nv12 = nv12[:, :width]
    nv12 = nv12.contiguous()
    nv12_nhwc = nv12.reshape(1, height * 3 // 2, width, 1)
    nv12_cv = cvcuda.as_tensor(nv12_nhwc, "NHWC")
    rgb_buf = torch.empty(1, height, width, 3, dtype=torch.uint8, device="cuda")
    rgb_cv = cvcuda.as_tensor(rgb_buf, "NHWC")
    cvcuda.cvtcolor_into(rgb_cv, nv12_cv, cvcuda.ColorConversion.YUV2RGB_NV12)
    return rgb_buf.squeeze(0)


def rgb_to_nv12(rgb):
    """RGB -> NV12 on GPU via cvcuda.

    Args:
        rgb: torch.Tensor (H, W, 3) uint8 on GPU.

    Returns:
        torch.Tensor (H*3//2, W) uint8 on GPU.
    """
    h, w = rgb.shape[:2]
    rgb_nhwc = rgb.unsqueeze(0)
    rgb_cv = cvcuda.as_tensor(rgb_nhwc, "NHWC")
    nv12_buf = torch.empty(1, h * 3 // 2, w, 1, dtype=torch.uint8, device="cuda")
    nv12_cv = cvcuda.as_tensor(nv12_buf, "NHWC")
    cvcuda.cvtcolor_into(nv12_cv, rgb_cv, cvcuda.ColorConversion.RGB2YUV_NV12)
    return nv12_buf.reshape(h * 3 // 2, w)


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
    _DECODE_BATCH_SIZE = 8

    def __init__(self):
        self._decoder = None
        self._encoder = None
        self._tmp_raw_path: str = ""
        self._raw_file = None
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

    def profile_stage_names(self) -> list[str]:
        return self._GPU_STAGES

    def gpu_stage_names(self) -> list[str]:
        return self._GPU_STAGES

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
        bitrate = crf_to_bitrate(
            self.args.crf,
            self.trt_model.output_w,
            self.trt_model.output_h,
            self.info["fps"],
        )
        self.log(f"Initializing NVENC ({self.args.codec}, {bitrate / 1e6:.1f} Mbps)...")

        raw_ext = ".h264" if self.args.codec == "h264" else ".hevc"
        tmp_fd, self._tmp_raw_path = tempfile.mkstemp(suffix=raw_ext)
        os.close(tmp_fd)

        self._encoder = nvc.CreateEncoder(
            self.trt_model.output_w,
            self.trt_model.output_h,
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
        in_h, in_w = self.trt_model.input_h, self.trt_model.input_w

        if self.profiler:
            self._process_frame_profiled(nv12_tensor, in_h, in_w)
        else:
            rgb = nv12_to_rgb(nv12_tensor, in_h, in_w)
            upscaled = self._infer_gpu(rgb)
            nv12_out = rgb_to_nv12(upscaled)
            self._write_bitstream(self._encoder.Encode(_patch_dlpack(nv12_out)))

    def _infer_gpu(self, rgb_hwc):
        """TRT inference entirely on GPU."""
        with torch.cuda.stream(self.trt_model.stream):
            inp = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            self.trt_model.gpu_input.copy_(inp)
            self.trt_model.context.execute_async_v3(stream_handle=self.trt_model.stream.cuda_stream)
            out = self.trt_model.gpu_output.squeeze(0).permute(1, 2, 0).contiguous()
            out = out.mul_(255.0).clamp_(0, 255).byte()
        self.trt_model.stream.synchronize()
        return out

    def _process_frame_profiled(self, nv12_tensor, in_h, in_w):
        """Inference with CUDA event profiling on default stream."""
        e0, e1, e2, e3, e4 = (torch.cuda.Event(enable_timing=True) for _ in range(5))
        cur_stream = torch.cuda.current_stream()

        e0.record(cur_stream)
        rgb = nv12_to_rgb(nv12_tensor, in_h, in_w)
        e1.record(cur_stream)

        inp = rgb.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        self.trt_model.gpu_input.copy_(inp)
        self.trt_model.context.execute_async_v3(stream_handle=cur_stream.cuda_stream)
        out = self.trt_model.gpu_output.squeeze(0).permute(1, 2, 0).contiguous()
        upscaled = out.mul_(255.0).clamp_(0, 255).byte()
        e2.record(cur_stream)

        nv12_out = rgb_to_nv12(upscaled)
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
