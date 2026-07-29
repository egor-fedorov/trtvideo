"""GPU-resident video upscaling via PyNvVideoCodec, CV-CUDA, and TensorRT."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from trtvideo.pipelines.base import BasePipeline
from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
from trtvideo.video.nvcodec.bitrate import auto_bitrate_from_source
from trtvideo.video.nvcodec.decoder import iter_locked_decode_frames
from trtvideo.video.nvcodec.encoder import (
    NvencCbrContract,
    format_nvenc_fps,
    gop_size_for_one_second,
)
from trtvideo.video.nvcodec.processor import NvcodecFrameProcessor
from trtvideo.video.output import (
    StreamingFfmpegMuxer,
    build_ffmpeg_streaming_mux_command,
)


class NvcodecPipeline(BasePipeline):
    """Pipeline: NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC."""

    DESCRIPTION = "TensorRT Video Upscaler"
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
        self._muxer: StreamingFfmpegMuxer | None = None
        self._frame_processor: NvcodecFrameProcessor | None = None
        self._color_spec_name = "bt709"
        super().__init__(args)

    def create_runtime(self) -> CvcudaTensorRTRuntime:
        """Create the torch-free TensorRT runtime."""
        return CvcudaTensorRTRuntime(
            self.engine_path,
            quiet=self.args.quiet,
            gpu_id=self.args.gpu_id,
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

    def _processor(self) -> NvcodecFrameProcessor:
        if self._frame_processor is None:
            raise RuntimeError("NVCodec frame processor is not initialized")
        return self._frame_processor

    def _nvc_module(self) -> Any:
        if self._nvc is None:
            raise RuntimeError("PyNvVideoCodec is not initialized")
        return self._nvc

    def _output_muxer(self) -> StreamingFfmpegMuxer:
        if self._muxer is None:
            raise RuntimeError("FFmpeg output muxer is not initialized")
        return self._muxer

    def validate_video_input(self, info) -> None:
        super().validate_video_input(info)
        if info.pix_fmt not in {"nv12", "yuv420p"}:
            print(
                "ERROR: upscale currently supports only 8-bit SDR NV12/yuv420p "
                f"input, got pix_fmt={info.pix_fmt or 'unknown'}. "
                "Transcode or tonemap the input to SDR yuv420p first."
            )
            sys.exit(1)

    def setup_decoder(self) -> None:
        self.log("Initializing NVDEC...")
        import PyNvVideoCodec as nvc

        self._nvc = nvc
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
        runtime = self._runtime()
        nvc = self._nvc_module()
        self._frame_processor = NvcodecFrameProcessor(
            runtime,
            color_spec_name=self._color_spec_name,
            limited_range=(
                self.normalized_color_metadata()["color_range"] != "pc"
            ),
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
        encoder_contract = NvencCbrContract(
            bitrate_bps=bitrate,
            gop_frames=gop_size,
            codec=self.args.codec,
        )

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
        mux_cmd = build_ffmpeg_streaming_mux_command(
            video_codec=self.args.codec,
            fps=self.info.fps_str,
            source_input_path=self.args.input,
            output_path=self.working_output_path(),
            preserve_chapters=self.args.max_frames <= 0,
            color_metadata_args=self.ffmpeg_color_metadata_args(),
            duration_args=self.ffmpeg_limited_duration_args(),
            faststart=os.path.splitext(self.args.output)[1].lower()
            in {".mp4", ".m4v", ".mov"},
        )
        self.log_verbose(f"Mux cmd: {' '.join(mux_cmd)}")
        self._muxer = StreamingFfmpegMuxer.start(mux_cmd)

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
        if bitstream:
            self._output_muxer().write(bytearray(bitstream))

    def _wrap_nv12(self, raw_frame) -> Any:
        return self._processor().wrap_nv12(raw_frame)

    def _preprocess(self, nv12_input: Any) -> None:
        self._processor().preprocess(nv12_input)

    def _postprocess(self) -> Any:
        return self._processor().postprocess()

    def process_frame(self, raw_frame) -> None:
        if self.profiler:
            self._process_frame_profiled(raw_frame)
            return
        if self._nvtx.enabled:
            self._process_frame_nvtx(raw_frame)
            return

        nv12_input = self._wrap_nv12(raw_frame)
        self._preprocess(nv12_input)
        self._processor().infer()
        nv12_output = self._postprocess()
        self._write_bitstream(self._encoder.Encode(nv12_output))

    def _process_frame_nvtx(self, raw_frame) -> None:
        nv12_input = self._wrap_nv12(raw_frame)
        with self._nvtx.range("trtvideo.nvcodec.nv12_to_rgb"):
            self._preprocess(nv12_input)
        with self._nvtx.range("trtvideo.nvcodec.tensorrt"):
            self._processor().infer()
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
        self._processor().infer()
        e2.record(runtime.stream)
        nv12_output = self._postprocess()
        e3.record(runtime.stream)
        self._write_bitstream(self._encoder.Encode(nv12_output))
        e4.record(runtime.stream)

        runtime.synchronize()
        self.profiler.commit(events)

    def finalize(self) -> None:
        """Flush NVENC and finish the streaming output container."""
        muxer = self._output_muxer()
        with self._nvtx.range("trtvideo.nvcodec.nvenc_flush"):
            self._write_bitstream(self._encoder.EndEncode())
        self._record_lifecycle_phase("encoder_drained")
        muxer.close_input()
        self._record_lifecycle_phase("mux_input_closed")

        self.log("\nFinalizing output container...")
        with self._nvtx.range("trtvideo.nvcodec.mux_finalize"):
            muxer.finish()
        self._record_lifecycle_phase("mux_completed")

    def cleanup(self) -> None:
        self._frame_processor = None
        if self._muxer is not None:
            self._muxer.abort()
            self._muxer = None
