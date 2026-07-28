"""GPU-resident video upscaling via PyNvVideoCodec, CV-CUDA, and TensorRT."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from trtvideo.pipelines.base import BasePipeline
from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
from trtvideo.video.bitrate import auto_bitrate_from_source
from trtvideo.video.cuda_array import nv12_nhwc_view
from trtvideo.video.decoder import iter_locked_decode_frames
from trtvideo.video.fps import format_nvenc_fps, gop_size_for_one_second
from trtvideo.video.nvenc import NvencCbrContract
from trtvideo.video.preservation import ffmpeg_preservation_args


@dataclass
class FrameBufferPool:
    """CV-CUDA buffers reused by every frame in one NVCodec job."""

    rgb_in_u8: Any
    rgb_in_float: Any
    rgb_out_float: Any
    rgb_out_u8: Any
    nv12_out: Any
    nv12_out_hwc: Any

    @classmethod
    def create(cls, runtime: CvcudaTensorRTRuntime) -> FrameBufferPool:
        cvcuda = runtime.cvcuda
        rgb_in_shape = (1, runtime.input_h, runtime.input_w, 3)
        rgb_out_shape = (1, runtime.output_h, runtime.output_w, 3)
        nv12_shape = (1, runtime.output_h * 3 // 2, runtime.output_w, 1)
        nv12_out = cvcuda.Tensor(nv12_shape, cvcuda.Type.U8, layout="NHWC")
        return cls(
            rgb_in_u8=cvcuda.Tensor(rgb_in_shape, cvcuda.Type.U8, layout="NHWC"),
            rgb_in_float=cvcuda.Tensor(
                rgb_in_shape,
                runtime.input_dtype,
                layout="NHWC",
            ),
            rgb_out_float=cvcuda.Tensor(
                rgb_out_shape,
                runtime.output_dtype,
                layout="NHWC",
            ),
            rgb_out_u8=cvcuda.Tensor(rgb_out_shape, cvcuda.Type.U8, layout="NHWC"),
            nv12_out=nv12_out,
            nv12_out_hwc=nv12_out.reshape(
                (runtime.output_h * 3 // 2, runtime.output_w, 1),
                layout="HWC",
            ),
        )


class NvcodecPipeline(BasePipeline):
    """Pipeline: NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC."""

    DESCRIPTION = "TensorRT Video Upscaler (NVDEC/NVENC backend)"
    BACKEND_NAME = "nvcodec"
    _DECODE_BATCH_SIZE = 8
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

    def __init__(self, args: argparse.Namespace):
        self._decoder: Any | None = None
        self._encoder: Any | None = None
        self._nvc: Any | None = None
        self._tmp_raw_path: str = ""
        self._raw_file: Any | None = None
        self._buffer_pool: FrameBufferPool | None = None
        self._color_spec_name = "bt709"
        self._color_spec: Any | None = None
        super().__init__(args)

    def create_runtime(self) -> CvcudaTensorRTRuntime:
        """Create the torch-free TensorRT runtime used by the NVCodec backend."""
        return CvcudaTensorRTRuntime(
            self.engine_path,
            quiet=self.args.quiet,
            gpu_id=self.args.gpu_id,
            use_cuda_graph=self.args.cuda_graph,
        )

    def profile_stage_names(self) -> list[str]:
        return self._GPU_STAGES

    def gpu_stage_names(self) -> list[str]:
        return self._GPU_STAGES

    def profile_stage_key_map(self) -> dict[str, str]:
        return self._PROFILE_STAGE_KEYS

    def _runtime(self) -> CvcudaTensorRTRuntime:
        runtime = self.require_runtime()
        if not isinstance(runtime, CvcudaTensorRTRuntime):
            raise RuntimeError("NVCodec pipeline requires CvcudaTensorRTRuntime")
        return runtime

    def _buffers(self) -> FrameBufferPool:
        if self._buffer_pool is None:
            raise RuntimeError("Frame buffer pool is not initialized")
        return self._buffer_pool

    def _nvc_module(self) -> Any:
        if self._nvc is None:
            raise RuntimeError("PyNvVideoCodec is not initialized")
        return self._nvc

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
        import PyNvVideoCodec as nvc

        self._nvc = nvc
        self._color_spec_name = self.cvcuda_color_spec_name()
        runtime = self._runtime()
        self._color_spec = getattr(runtime.cvcuda.ColorSpec, self._color_spec_name.upper())
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
        runtime = self._runtime()
        nvc = self._nvc_module()
        self._buffer_pool = FrameBufferPool.create(runtime)
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
        encoder_contract = NvencCbrContract(
            bitrate_bps=bitrate,
            gop_frames=gop_size,
            codec=self.args.codec,
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
            cudastream=runtime.stream_handle,
            fps=encoder_fps,
            **encoder_contract.pynvcodec_options(),
        )
        self._raw_file = open(self._tmp_raw_path, "wb")

    def _resolve_bitrate(self, runtime: CvcudaTensorRTRuntime) -> int:
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
        """Yield NV12 surfaces while respecting ThreadedDecoder buffer ownership."""
        runtime = self._runtime()
        fetch_batch = self._decoder.get_batch_frames
        if self._nvtx.enabled:

            def annotated_fetch_batch(batch_size: int):
                with self._nvtx.range("trtvideo.nvcodec.decode_batch"):
                    return self._decoder.get_batch_frames(batch_size)

            fetch_batch = annotated_fetch_batch

        yield from iter_locked_decode_frames(
            fetch_batch,
            batch_size=self._DECODE_BATCH_SIZE,
            release_batch=runtime.synchronize,
        )

    def _write_bitstream(self, bitstream) -> None:
        if bitstream and self._raw_file:
            self._raw_file.write(bytearray(bitstream))

    def _wrap_nv12(self, raw_frame) -> Any:
        runtime = self._runtime()
        source = runtime.cvcuda.as_tensor(raw_frame)
        view = nv12_nhwc_view(
            source,
            height=runtime.input_h,
            width=runtime.input_w,
        )
        return runtime.cvcuda.as_tensor(view, layout="NHWC")

    def _preprocess(self, nv12_input: Any) -> None:
        runtime = self._runtime()
        buffers = self._buffers()
        cvcuda = runtime.cvcuda
        cvcuda.advcvtcolor_into(
            buffers.rgb_in_u8,
            nv12_input,
            cvcuda.ColorConversion.YUV2RGB_NV12,
            self._color_spec,
            stream=runtime.stream,
        )
        cvcuda.convertto_into(
            buffers.rgb_in_float,
            buffers.rgb_in_u8,
            scale=1.0 / 255.0,
            stream=runtime.stream,
        )
        cvcuda.reformat_into(
            runtime.gpu_input,
            buffers.rgb_in_float,
            stream=runtime.stream,
        )

    def _postprocess(self) -> Any:
        runtime = self._runtime()
        buffers = self._buffers()
        cvcuda = runtime.cvcuda
        cvcuda.reformat_into(
            buffers.rgb_out_float,
            runtime.gpu_output,
            stream=runtime.stream,
        )
        cvcuda.convertto_into(
            buffers.rgb_out_u8,
            buffers.rgb_out_float,
            scale=255.0,
            stream=runtime.stream,
        )
        cvcuda.advcvtcolor_into(
            buffers.nv12_out,
            buffers.rgb_out_u8,
            cvcuda.ColorConversion.RGB2YUV_NV12,
            self._color_spec,
            stream=runtime.stream,
        )
        return buffers.nv12_out_hwc

    def process_frame(self, raw_frame) -> None:
        if self.profiler:
            self._process_frame_profiled(raw_frame)
            return
        if self._nvtx.enabled:
            self._process_frame_nvtx(raw_frame)
            return

        runtime = self._runtime()
        nv12_input = self._wrap_nv12(raw_frame)
        self._preprocess(nv12_input)
        runtime.execute()
        nv12_output = self._postprocess()
        self._write_bitstream(self._encoder.Encode(nv12_output))

    def _process_frame_nvtx(self, raw_frame) -> None:
        runtime = self._runtime()
        nv12_input = self._wrap_nv12(raw_frame)
        with self._nvtx.range("trtvideo.nvcodec.nv12_to_rgb"):
            self._preprocess(nv12_input)
        with self._nvtx.range("trtvideo.nvcodec.tensorrt"):
            runtime.execute()
        with self._nvtx.range("trtvideo.nvcodec.rgb_to_nv12"):
            nv12_output = self._postprocess()
        with self._nvtx.range("trtvideo.nvcodec.nvenc_encode"):
            self._write_bitstream(self._encoder.Encode(nv12_output))

    def _process_frame_profiled(self, raw_frame) -> None:
        runtime = self._runtime()
        events = tuple(runtime.create_timing_event() for _ in range(5))
        e0, e1, e2, e3, e4 = events
        nv12_input = self._wrap_nv12(raw_frame)

        e0.record(runtime.stream)
        self._preprocess(nv12_input)
        e1.record(runtime.stream)
        runtime.execute()
        e2.record(runtime.stream)
        nv12_output = self._postprocess()
        e3.record(runtime.stream)
        self._write_bitstream(self._encoder.Encode(nv12_output))
        e4.record(runtime.stream)

        runtime.synchronize()
        self.profiler.commit(events)

    def finalize(self) -> None:
        """Flush NVENC and mux the generated video with preserved source media."""
        with self._nvtx.range("trtvideo.nvcodec.nvenc_flush"):
            self._write_bitstream(self._encoder.EndEncode())
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None
        self._record_lifecycle_phase("encoder_flushed")

        self.log("\nMuxing output container...")
        faststart_args = (
            ["-movflags", "+faststart"]
            if os.path.splitext(self.args.output)[1].lower() in {".mp4", ".m4v", ".mov"}
            else []
        )
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
            *ffmpeg_preservation_args(preserve_chapters=self.args.max_frames <= 0),
            *self.ffmpeg_color_metadata_args(),
            *self.ffmpeg_limited_duration_args(),
            *faststart_args,
            self.working_output_path(),
        ]
        self.log_verbose(f"Mux cmd: {' '.join(mux_cmd)}")
        with self._nvtx.range("trtvideo.nvcodec.mux"):
            result = subprocess.run(mux_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            details = result.stderr.strip() or "ffmpeg returned no error details"
            raise RuntimeError(f"ffmpeg mux failed:\n{details}")
        self._record_lifecycle_phase("mux_completed")

    def cleanup(self) -> None:
        self._buffer_pool = None
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None
        if self._tmp_raw_path and os.path.exists(self._tmp_raw_path):
            os.unlink(self._tmp_raw_path)
