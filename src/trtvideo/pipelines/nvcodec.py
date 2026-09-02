"""GPU-resident video processing via PyNvVideoCodec, CV-CUDA, and TensorRT."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from trtvideo.benchmarking.lifecycle import LifecycleRecorder
from trtvideo.diagnostics.nvtx import NvtxAnnotator
from trtvideo.diagnostics.profiling import ProfileCollector, write_profile_report
from trtvideo.pipelines.config import PipelineError, ProcessConfig
from trtvideo.pipelines.reporting import (
    PROCESS_RESULT_DOCUMENT_TYPE,
    PROCESS_RESULT_SCHEMA_VERSION,
    JsonLinesWriter,
    ProcessReportingError,
    ProcessTimingSummary,
    build_progress_event,
    render_progress,
    summarize_process_timing,
    write_json_document,
)
from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
from trtvideo.video.color import ColorContractError, SdrColorContract
from trtvideo.video.frames import iter_limited_frames
from trtvideo.video.metadata import VideoMetadata
from trtvideo.video.nvcodec.bitrate import auto_bitrate_from_source
from trtvideo.video.nvcodec.decoder import iter_locked_decode_frames
from trtvideo.video.nvcodec.encoder import (
    NvencCbrContract,
    format_nvenc_fps,
    gop_size_for_one_second,
)
from trtvideo.video.nvcodec.frame_processor import NvcodecFrameProcessor
from trtvideo.video.output import (
    AtomicOutputTransaction,
    MediaPreservationError,
    StreamingFfmpegMuxer,
    build_ffmpeg_streaming_mux_command,
)
from trtvideo.video.probe import VideoProbeError, probe_video


class NvcodecPipeline:
    """Orchestrate NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC."""

    _DECODE_BATCH_SIZE = 8
    _GPU_STAGES = [
        "NV12→RGB (cvcuda)",
        "TRT inference",
        "RGB→NV12 (cvcuda)",
        "NVENC encode",
    ]
    _PROFILE_STAGE_KEYS = {
        "NV12→RGB (cvcuda)": "nv12_to_rgb",
        "TRT inference": "trt",
        "RGB→NV12 (cvcuda)": "rgb_to_nv12",
        "NVENC encode": "encode",
    }

    def __init__(self, config: ProcessConfig):
        self.config = config
        self._info: VideoMetadata | None = None
        self._color: SdrColorContract | None = None
        self._runtime_engine: CvcudaTensorRTRuntime | None = None
        self._decoder: Any | None = None
        self._encoder: Any | None = None
        self._nvc: Any | None = None
        self._muxer: StreamingFfmpegMuxer | None = None
        self._frame_processor: NvcodecFrameProcessor | None = None
        self._profiler: ProfileCollector | None = None
        self._working_output_path: Path | None = None
        self._total_frames: int | None = None
        self._progress_writer: JsonLinesWriter | None = None
        self._nvtx = NvtxAnnotator.from_environment()
        self._lifecycle = LifecycleRecorder(config.benchmark_lifecycle_path)
        self._lifecycle.mark_phase("pipeline_created")

    def run(self) -> None:
        """Process the configured video and atomically publish its output."""
        try:
            with JsonLinesWriter(self.config.progress_jsonl_path) as progress_writer:
                self._progress_writer = progress_writer
                try:
                    self._run()
                finally:
                    self._progress_writer = None
        except (
            ColorContractError,
            MediaPreservationError,
            ProcessReportingError,
            VideoProbeError,
        ) as exc:
            raise PipelineError(str(exc)) from exc

    def _run(self) -> None:
        pipeline_started = time.perf_counter()
        self._validate_paths()
        info = probe_video(str(self.config.input_path))
        self._info = info
        self._lifecycle.mark_phase("video_probed")
        self._log_input(info)
        self._color = SdrColorContract.from_video_info(info)
        self._validate_video_input(info)
        self._total_frames = self.config.max_frames if self.config.max_frames > 0 else None
        if self._total_frames is not None and info.nb_frames > 0:
            self._total_frames = min(self._total_frames, info.nb_frames)
        if self._total_frames is None and info.nb_frames > 0:
            self._total_frames = info.nb_frames

        frame_times: list[float] = []
        transaction = AtomicOutputTransaction(
            str(self.config.input_path),
            str(self.config.output_path),
            preserve_chapters=self.config.max_frames <= 0,
        )
        try:
            with transaction as working_output_path:
                self._working_output_path = working_output_path
                self._lifecycle.mark_phase("preservation_preflight_completed")
                try:
                    frame_loop_sec, active_pipeline_sec = self._execute_pipeline(frame_times)
                finally:
                    self._cleanup()
                    self._lifecycle.mark_phase("cleanup_completed")
            self._lifecycle.mark_phase("output_committed")
        finally:
            self._working_output_path = None

        pipeline_wall_sec = time.perf_counter() - pipeline_started
        self._lifecycle.mark_phase("reporting_started")
        self._lifecycle.write(len(frame_times))
        self._report_profile(frame_times, active_pipeline_sec)
        summary = summarize_process_timing(
            frame_times,
            warmup_frames=self.config.warmup_frames,
            frame_loop_sec=frame_loop_sec,
            active_pipeline_sec=active_pipeline_sec,
            pipeline_wall_sec=pipeline_wall_sec,
        )
        self._print_stats(summary)
        self._write_result(summary)

    def _execute_pipeline(self, frame_times: list[float]) -> tuple[float, float]:
        info = self._video_info()
        with self._nvtx.range("trtvideo.initialization"):
            self._log("\nInitializing TensorRT...")
            self._runtime_engine = CvcudaTensorRTRuntime(
                str(self.config.engine_path),
                quiet=self.config.quiet,
                gpu_id=self.config.gpu_id,
                log_stream=sys.stderr,
            )
            runtime = self._runtime()
            if info.width != runtime.input_w or info.height != runtime.input_h:
                raise PipelineError(
                    f"Video {info.width}x{info.height} does not match engine "
                    f"{runtime.input_w}x{runtime.input_h}"
                )
            self._lifecycle.mark_phase("runtime_initialized")

            if self.config.profile or self.config.profile_json_path:
                self._profiler = ProfileCollector(
                    self._GPU_STAGES,
                    gpu_stages=self._GPU_STAGES,
                    synchronize=runtime.synchronize,
                    skip_warmup=self.config.warmup_frames,
                )

            self._setup_decoder()
            self._lifecycle.mark_phase("decoder_initialized")
            self._setup_encoder()
            self._lifecycle.mark_phase("encoder_initialized")

            if self._total_frames is None:
                self._log("\nProcessing: all available frames")
            else:
                self._log(f"\nProcessing: {self._total_frames} frames")
            self._log(
                f"Output: {self.config.output_path} ({runtime.output_w}x{runtime.output_h})\n"
            )

        wall_start = time.perf_counter()
        with self._nvtx.range("trtvideo.frame_loop"):
            self._run_frame_loop(frame_times, frame_loop_started=wall_start)
        frame_loop_sec = time.perf_counter() - wall_start
        self._lifecycle.mark_phase("frame_loop_completed")
        with self._nvtx.range("trtvideo.finalize"):
            self._finalize()
        self._lifecycle.mark_phase("pipeline_finalized")
        return frame_loop_sec, time.perf_counter() - wall_start

    def _validate_paths(self) -> None:
        if not self.config.engine_path.exists():
            raise PipelineError(f"Engine not found: {self.config.engine_path}")
        if not self.config.input_path.exists():
            raise PipelineError(f"Video not found: {self.config.input_path}")

    def _validate_video_input(self, info: VideoMetadata) -> None:
        if info.pix_fmt not in {"nv12", "yuv420p"}:
            raise PipelineError(
                "trtvideo currently supports only 8-bit SDR NV12/yuv420p input, "
                f"got pix_fmt={info.pix_fmt or 'unknown'}. "
                "Transcode or tonemap the input to SDR yuv420p first."
            )

    def _log_input(self, info: VideoMetadata) -> None:
        self._log(
            f"Input video: {info.width}x{info.height}, {info.fps:.2f} fps, {info.nb_frames} frames"
        )
        self._log(
            "Input color: "
            f"pix_fmt={info.pix_fmt or 'unknown'}, "
            f"range={info.color_range or 'unknown'}, "
            f"space={info.color_space or 'unknown'}, "
            f"transfer={info.color_transfer or 'unknown'}, "
            f"primaries={info.color_primaries or 'unknown'}"
        )

    def _setup_decoder(self) -> None:
        self._log("Initializing NVDEC...")
        import PyNvVideoCodec as nvc

        self._nvc = nvc
        color_spec_name = self._color_contract().cvcuda_spec_name
        self._log_verbose(f"CV-CUDA color spec: {color_spec_name}")
        self._decoder = nvc.ThreadedDecoder(
            enc_file_path=str(self.config.input_path),
            buffer_size=self._DECODE_BATCH_SIZE,
            gpu_id=self.config.gpu_id,
            cuda_context=0,
            cuda_stream=0,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
        )

    def _setup_encoder(self) -> None:
        runtime = self._runtime()
        info = self._video_info()
        color = self._color_contract()
        nvc = self._nvc_module()
        self._frame_processor = NvcodecFrameProcessor(
            runtime,
            color_spec_name=color.cvcuda_spec_name,
            limited_range=color.limited_range,
        )
        bitrate = self._resolve_bitrate(runtime)
        try:
            encoder_fps = format_nvenc_fps(info.fps_str)
            gop_size = gop_size_for_one_second(info.fps_str)
        except ValueError as exc:
            raise PipelineError(f"Unsupported input FPS for NVENC: {exc}") from exc

        self._log(
            f"Initializing NVENC ({self.config.codec}, {bitrate / 1e6:.1f} Mbps, "
            f"{color.cvcuda_spec_name}, fps={encoder_fps}, gop={gop_size}, bframes=0)..."
        )
        encoder_contract = NvencCbrContract(
            bitrate_bps=bitrate,
            gop_frames=gop_size,
            codec=self.config.codec,
        )
        self._encoder = nvc.CreateEncoder(
            runtime.output_w,
            runtime.output_h,
            "NV12",
            False,
            gpu_id=self.config.gpu_id,
            cudastream=runtime.stream_handle,
            fps=encoder_fps,
            **encoder_contract.pynvcodec_options(),
        )
        mux_command = build_ffmpeg_streaming_mux_command(
            video_codec=self.config.codec,
            fps=info.fps_str,
            source_input_path=str(self.config.input_path),
            output_path=str(self._working_output()),
            preserve_chapters=self.config.max_frames <= 0,
            color_metadata_args=color.ffmpeg_args(),
            duration_args=self._limited_duration_args(),
            faststart=self.config.output_path.suffix.lower() in {".mp4", ".m4v", ".mov"},
        )
        self._log_verbose(f"Mux cmd: {' '.join(mux_command)}")
        self._muxer = StreamingFfmpegMuxer.start(mux_command)

    def _resolve_bitrate(self, runtime: CvcudaTensorRTRuntime) -> int:
        info = self._video_info()
        if self.config.bitrate_mbps is not None:
            bitrate = int(self.config.bitrate_mbps * 1_000_000)
            self._log(f"NVENC bitrate: manual {bitrate / 1e6:.1f} Mbps")
            return bitrate

        source_bitrate = info.video_bit_rate or info.container_bit_rate
        if source_bitrate:
            bitrate, pixel_ratio, fps_ratio = auto_bitrate_from_source(
                source_bitrate=source_bitrate,
                input_w=runtime.input_w,
                input_h=runtime.input_h,
                output_w=runtime.output_w,
                output_h=runtime.output_h,
                input_fps=info.fps,
                output_fps=info.fps,
            )
            source = "video stream" if info.video_bit_rate else "container"
            self._log(
                "NVENC bitrate auto: "
                f"source={source_bitrate / 1e6:.1f} Mbps ({source}), "
                f"pixel_ratio={pixel_ratio:.2f}, fps_ratio={fps_ratio:.2f}, "
                f"target={bitrate / 1e6:.1f} Mbps"
            )
            return bitrate

        raise PipelineError(
            "nvcodec auto bitrate requires source video bitrate metadata. "
            "Pass --bitrate-mbps explicitly."
        )

    def _decode_frames(self):
        """Yield NV12 surfaces while respecting ThreadedDecoder ownership."""
        runtime = self._runtime()
        decoder = self._decoder_instance()
        fetch_batch = decoder.get_batch_frames
        if self._nvtx.enabled:

            def annotated_fetch_batch(batch_size: int):
                with self._nvtx.range("trtvideo.nvcodec.decode_batch"):
                    return decoder.get_batch_frames(batch_size)

            fetch_batch = annotated_fetch_batch

        yield from iter_locked_decode_frames(
            fetch_batch,
            batch_size=self._DECODE_BATCH_SIZE,
            release_batch=runtime.synchronize,
        )

    def _run_frame_loop(
        self,
        frame_times: list[float],
        *,
        frame_loop_started: float,
    ) -> None:
        frame_index = 0
        last_reported_frame = 0
        last_reported_at = frame_loop_started
        progress_enabled = not self.config.quiet or (
            self._progress_writer is not None and self._progress_writer.enabled
        )
        for raw_frame in iter_limited_frames(
            self._decode_frames(),
            limit=self.config.max_frames,
        ):
            started = time.perf_counter()
            self._process_frame(raw_frame)
            completed = time.perf_counter()
            self._lifecycle.mark_frame_completed()

            frame_time = completed - started
            frame_times.append(frame_time)
            frame_index += 1
            if progress_enabled and (
                frame_index % self.config.log_interval == 0
                or frame_index == 1
                or frame_index == self._total_frames
            ):
                reported_at = time.perf_counter()
                self._report_progress(
                    frame_index=frame_index,
                    frame_loop_elapsed_sec=reported_at - frame_loop_started,
                    last_frame_processing_sec=frame_times[-1],
                    window_frames=frame_index - last_reported_frame,
                    window_elapsed_sec=reported_at - last_reported_at,
                )
                last_reported_frame = frame_index
                last_reported_at = reported_at
        if progress_enabled and frame_index > 0 and last_reported_frame != frame_index:
            reported_at = time.perf_counter()
            self._report_progress(
                frame_index=frame_index,
                frame_loop_elapsed_sec=reported_at - frame_loop_started,
                last_frame_processing_sec=frame_times[-1],
                window_frames=frame_index - last_reported_frame,
                window_elapsed_sec=reported_at - last_reported_at,
            )

    def _report_progress(
        self,
        *,
        frame_index: int,
        frame_loop_elapsed_sec: float,
        last_frame_processing_sec: float,
        window_frames: int,
        window_elapsed_sec: float,
    ) -> None:
        event = build_progress_event(
            processed_frames=frame_index,
            total_frames=self._total_frames,
            frame_loop_elapsed_sec=frame_loop_elapsed_sec,
            last_frame_processing_sec=last_frame_processing_sec,
            window_frames=window_frames,
            window_elapsed_sec=window_elapsed_sec,
        )
        if self._progress_writer is not None:
            self._progress_writer.write(event)
        if not self.config.quiet:
            self._log(render_progress(event))

    def _process_frame(self, raw_frame: Any) -> None:
        if self._profiler is not None:
            self._process_frame_profiled(raw_frame)
            return
        if self._nvtx.enabled:
            self._process_frame_nvtx(raw_frame)
            return

        processor = self._processor()
        encoder = self._encoder_instance()
        nv12_input = processor.wrap_nv12(raw_frame)
        processor.preprocess(nv12_input)
        processor.infer()
        nv12_output = processor.postprocess()
        self._write_bitstream(encoder.Encode(nv12_output))

    def _process_frame_nvtx(self, raw_frame: Any) -> None:
        processor = self._processor()
        encoder = self._encoder_instance()
        nv12_input = processor.wrap_nv12(raw_frame)
        with self._nvtx.range("trtvideo.nvcodec.nv12_to_rgb"):
            processor.preprocess(nv12_input)
        with self._nvtx.range("trtvideo.nvcodec.tensorrt"):
            processor.infer()
        with self._nvtx.range("trtvideo.nvcodec.rgb_to_nv12"):
            nv12_output = processor.postprocess()
        with self._nvtx.range("trtvideo.nvcodec.nvenc_encode"):
            self._write_bitstream(encoder.Encode(nv12_output))

    def _process_frame_profiled(self, raw_frame: Any) -> None:
        runtime = self._runtime()
        processor = self._processor()
        encoder = self._encoder_instance()
        profiler = self._profile_collector()
        events = tuple(runtime.create_timing_event() for _ in range(5))
        e0, e1, e2, e3, e4 = events
        nv12_input = processor.wrap_nv12(raw_frame)

        e0.record(runtime.stream)
        processor.preprocess(nv12_input)
        e1.record(runtime.stream)
        processor.infer()
        e2.record(runtime.stream)
        nv12_output = processor.postprocess()
        e3.record(runtime.stream)
        self._write_bitstream(encoder.Encode(nv12_output))
        e4.record(runtime.stream)

        runtime.synchronize()
        profiler.commit(events)

    def _finalize(self) -> None:
        muxer = self._output_muxer()
        with self._nvtx.range("trtvideo.nvcodec.nvenc_flush"):
            self._write_bitstream(self._encoder_instance().EndEncode())
        self._lifecycle.mark_phase("encoder_drained")
        muxer.close_input()
        self._lifecycle.mark_phase("mux_input_closed")

        self._log("\nFinalizing output container...")
        with self._nvtx.range("trtvideo.nvcodec.mux_finalize"):
            muxer.finish()
        self._lifecycle.mark_phase("mux_completed")

    def _cleanup(self) -> None:
        self._frame_processor = None
        if self._muxer is not None:
            self._muxer.abort()
            self._muxer = None

    def _report_profile(self, frame_times: list[float], wall_total: float) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        runtime = self._runtime()
        info = self._video_info()
        if self.config.profile and profiler.committed_count > 0:
            profiler.print_table(
                runtime.input_w,
                runtime.input_h,
                runtime.output_w,
                runtime.output_h,
                frame_times,
                output=sys.stderr,
            )
        if self.config.profile_json_path is not None:
            write_profile_report(
                self.config.profile_json_path,
                collector=profiler,
                runtime=runtime,
                video_info=info,
                engine_path=self.config.engine_path,
                input_path=self.config.input_path,
                media_output_path=self.config.output_path,
                frame_times=frame_times,
                wall_total_sec=wall_total,
                stage_key_map=self._PROFILE_STAGE_KEYS,
            )

    def _limited_duration_args(self) -> list[str]:
        info = self._video_info()
        if self.config.max_frames <= 0 or info.fps <= 0:
            return []
        duration_sec = self.config.max_frames / info.fps
        return ["-t", f"{duration_sec:.6f}"]

    def _print_stats(self, summary: ProcessTimingSummary) -> None:
        if self.config.quiet or summary.processed_frames == 0:
            return

        print(f"\n{'=' * 50}", file=sys.stderr)
        print(f"Frames processed: {summary.processed_frames}", file=sys.stderr)
        print(
            f"Frame loop:       {summary.frame_loop_sec:.1f}s ({summary.frame_loop_fps:.1f} FPS)",
            file=sys.stderr,
        )
        print(
            f"Active pipeline:  {summary.active_pipeline_sec:.1f}s "
            f"({summary.active_pipeline_fps:.1f} FPS)",
            file=sys.stderr,
        )
        print(
            f"Pipeline wall:    {summary.pipeline_wall_sec:.1f}s "
            f"({summary.pipeline_wall_fps:.1f} FPS)",
            file=sys.stderr,
        )
        if self.config.verbose:
            print(
                "Frame body:       "
                f"avg {summary.average_frame_processing_sec:.3f}s, "
                f"post-warmup {summary.post_warmup_average_frame_processing_sec:.3f}s, "
                f"min/max {summary.min_frame_processing_sec:.3f}s / "
                f"{summary.max_frame_processing_sec:.3f}s "
                f"({summary.warmup_frames_excluded} excluded)",
                file=sys.stderr,
            )
        print(f"Output file:      {self.config.output_path}", file=sys.stderr)
        print(f"{'=' * 50}", file=sys.stderr)

    def _write_result(self, summary: ProcessTimingSummary) -> None:
        destination = self.config.result_json_path
        if destination is None:
            return
        runtime = self._runtime()
        info = self._video_info()
        report: dict[str, object] = {
            "document_type": PROCESS_RESULT_DOCUMENT_TYPE,
            "schema_version": PROCESS_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "engine": {"path": str(self.config.engine_path)},
            "gpu": {
                "id": self.config.gpu_id,
                "name": runtime.gpu_name,
            },
            "input": {
                "path": str(self.config.input_path),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frame_rate": info.fps_str,
                "frame_count": info.nb_frames,
            },
            "output": {
                "path": str(self.config.output_path),
                "width": runtime.output_w,
                "height": runtime.output_h,
                "fps": info.fps,
                "frame_rate": info.fps_str,
                "frame_count": summary.processed_frames,
                "codec": self.config.codec,
            },
            "request": {
                "max_frames": self.config.max_frames or None,
                "warmup_frames": self.config.warmup_frames,
                "log_interval": self.config.log_interval,
            },
            "processed_frames": summary.processed_frames,
            "timing": summary.as_json(),
        }
        write_json_document(destination, report)

    def _write_bitstream(self, bitstream: Any) -> None:
        if bitstream:
            self._output_muxer().write(bytearray(bitstream))

    def _video_info(self) -> VideoMetadata:
        if self._info is None:
            raise RuntimeError("Video metadata is not initialized")
        return self._info

    def _color_contract(self) -> SdrColorContract:
        if self._color is None:
            raise RuntimeError("Color contract is not initialized")
        return self._color

    def _runtime(self) -> CvcudaTensorRTRuntime:
        if self._runtime_engine is None:
            raise RuntimeError("TensorRT runtime is not initialized")
        return self._runtime_engine

    def _processor(self) -> NvcodecFrameProcessor:
        if self._frame_processor is None:
            raise RuntimeError("NVCodec frame processor is not initialized")
        return self._frame_processor

    def _decoder_instance(self) -> Any:
        if self._decoder is None:
            raise RuntimeError("NVDEC decoder is not initialized")
        return self._decoder

    def _encoder_instance(self) -> Any:
        if self._encoder is None:
            raise RuntimeError("NVENC encoder is not initialized")
        return self._encoder

    def _nvc_module(self) -> Any:
        if self._nvc is None:
            raise RuntimeError("PyNvVideoCodec is not initialized")
        return self._nvc

    def _output_muxer(self) -> StreamingFfmpegMuxer:
        if self._muxer is None:
            raise RuntimeError("FFmpeg output muxer is not initialized")
        return self._muxer

    def _working_output(self) -> Path:
        if self._working_output_path is None:
            raise RuntimeError("Working output path is not initialized")
        return self._working_output_path

    def _profile_collector(self) -> ProfileCollector:
        if self._profiler is None:
            raise RuntimeError("Stage profiler is not initialized")
        return self._profiler

    def _log(self, *values: object) -> None:
        if not self.config.quiet:
            print(*values, file=sys.stderr)

    def _log_verbose(self, *values: object) -> None:
        if self.config.verbose:
            print(*values, file=sys.stderr)
