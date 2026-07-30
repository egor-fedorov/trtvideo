"""Base pipeline with template method pattern for the frame processing loop."""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from trtvideo.benchmarking.lifecycle import FrameLifecycleMarkers, write_frame_markers
from trtvideo.diagnostics.nvtx import NvtxAnnotator
from trtvideo.runtime import RuntimeEngine
from trtvideo.video.frames import iter_limited_frames
from trtvideo.video.output import (
    MediaPreservationError,
    commit_atomic_output,
    create_staging_output,
    preflight_output_container,
)
from trtvideo.video.probe import VideoInfo, probe_video

if TYPE_CHECKING:
    from trtvideo.diagnostics.profiling import ProfileCollector

_UNKNOWN_COLOR_VALUES = {None, "", "unknown", "reserved"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def _known_color_value(value: str | None) -> bool:
    return value not in _UNKNOWN_COLOR_VALUES


def _color_or_default(value: str | None, default: str) -> str:
    return value if _known_color_value(value) and value is not None else default


class BasePipeline(ABC):
    """Base class for video upscaling pipeline.

    Template method: run() controls the full cycle.
    Subclasses implement hooks: setup_decoder, setup_encoder, decode_frames,
    process_frame, finalize, cleanup.
    """

    DESCRIPTION: str = "TensorRT Video Upscaler"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.info = VideoInfo(width=0, height=0, fps=0.0, fps_str="0/1", nb_frames=0)
        self.runtime: RuntimeEngine | None = None
        self.engine_path: str = ""
        self.total_frames: int = 0
        self.profiler: ProfileCollector | None = None
        self._first_frame_completed_ns: int | None = None
        self._last_frame_completed_ns: int | None = None
        self._working_output_path: Path | None = None
        self._lifecycle_phase_completed_ns: dict[str, int] = {}
        self._nvtx = NvtxAnnotator.from_environment()
        self._record_lifecycle_phase("pipeline_created")

    # --- Logging helpers ---

    def log(self, *a, **kw):
        if not self.args.quiet:
            print(*a, **kw)

    def log_verbose(self, *a, **kw):
        if self.args.verbose:
            print(*a, **kw)

    def require_runtime(self) -> RuntimeEngine:
        """Return initialized runtime or fail on invalid lifecycle usage."""
        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")
        return self.runtime

    def working_output_path(self) -> str:
        """Return the private output path used until the pipeline succeeds."""
        if self._working_output_path is None:
            raise RuntimeError("Working output path is not initialized")
        return str(self._working_output_path)

    def ffmpeg_limited_duration_args(self) -> list[str]:
        """Return ffmpeg output args that keep audio aligned with --max-frames."""
        if self.args.max_frames <= 0 or self.info.fps <= 0:
            return []
        duration_sec = self.args.max_frames / self.info.fps
        return ["-t", f"{duration_sec:.6f}"]

    def default_sdr_colorspace(self) -> str:
        """Infer a safe SDR YUV matrix when source metadata is absent."""
        return "bt709" if self.info.width >= 1280 or self.info.height >= 720 else "smpte170m"

    def normalized_color_metadata(self) -> dict[str, str]:
        """Return explicit SDR color metadata for ffmpeg output tagging."""
        default_colorspace = self.default_sdr_colorspace()
        colorspace = _color_or_default(self.info.color_space, default_colorspace)
        return {
            "color_range": _color_or_default(self.info.color_range, "tv"),
            "colorspace": colorspace,
            "color_trc": _color_or_default(self.info.color_transfer, default_colorspace),
            "color_primaries": _color_or_default(self.info.color_primaries, default_colorspace),
        }

    def ffmpeg_color_metadata_args(self) -> list[str]:
        """Return ffmpeg args that avoid unknown color tags in encoded outputs."""
        metadata = self.normalized_color_metadata()
        return [
            "-color_range",
            metadata["color_range"],
            "-colorspace",
            metadata["colorspace"],
            "-color_trc",
            metadata["color_trc"],
            "-color_primaries",
            metadata["color_primaries"],
        ]

    def cvcuda_color_spec_name(self) -> str:
        """Map video metadata to the color specs supported by CV-CUDA AdvCvtColor."""
        colorspace = self.normalized_color_metadata()["colorspace"]
        if colorspace in {"bt2020nc", "bt2020c"}:
            return "bt2020"
        if colorspace in {"smpte170m", "bt470bg", "bt470m"}:
            return "bt601"
        return "bt709"

    def validate_video_input(self, info: VideoInfo) -> None:
        """Fail fast for inputs outside the current SDR model contract."""
        if info.color_transfer in _HDR_TRANSFERS:
            print(
                "ERROR: HDR input is not supported by the current SDR RGB model contract: "
                f"color_transfer={info.color_transfer}. Convert/tonemap to SDR first."
            )
            sys.exit(1)

    def profile_stage_key_map(self) -> dict[str, str]:
        """Map human-readable profile stage names to stable JSON keys."""
        return {}

    # --- Abstract hooks ---

    @abstractmethod
    def create_runtime(self) -> RuntimeEngine:
        """Create the pipeline runtime."""

    @abstractmethod
    def profile_stage_names(self) -> list[str]:
        """List of all stage names for profiling."""

    @abstractmethod
    def gpu_stage_names(self) -> list[str]:
        """List of GPU stage names (measured via CUDA events)."""

    @abstractmethod
    def setup_decoder(self) -> None:
        """Initialize the decoder."""

    @abstractmethod
    def setup_encoder(self) -> None:
        """Initialize the encoder."""

    @abstractmethod
    def decode_frames(self):
        """Generator of decoded frames."""

    @abstractmethod
    def process_frame(self, raw_frame) -> None:
        """Process one frame: preprocess -> TRT -> postprocess -> encode."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush encoder, mux, finalize."""

    def after_frame(self) -> None:
        """Hook after frame timing (e.g., write to pipe)."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (called in finally)."""

    # --- Template method ---

    def run(self) -> None:
        """Full pipeline: init -> frame loop -> stats."""
        args = self.args

        if args.engine and not os.path.exists(args.engine):
            print(f"ERROR: Engine not found: {args.engine}")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"ERROR: Video not found: {args.input}")
            sys.exit(1)

        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_upscaled{ext}"

        self.info = probe_video(args.input)
        self._record_lifecycle_phase("video_probed")
        info = self.info
        self.log(
            f"Input video: {info.width}x{info.height}, "
            f"{info.fps:.2f} fps, {info.nb_frames} frames"
        )
        self.log(
            "Input color: "
            f"pix_fmt={info.pix_fmt or 'unknown'}, "
            f"range={info.color_range or 'unknown'}, "
            f"space={info.color_space or 'unknown'}, "
            f"transfer={info.color_transfer or 'unknown'}, "
            f"primaries={info.color_primaries or 'unknown'}"
        )
        self.validate_video_input(info)
        self.total_frames = args.max_frames if args.max_frames > 0 else info.nb_frames
        self.engine_path = args.engine

        try:
            preflight_output_container(
                args.input,
                args.output,
                preserve_chapters=args.max_frames <= 0,
            )
            working_output_path = create_staging_output(args.output)
        except MediaPreservationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        self._working_output_path = working_output_path
        self._record_lifecycle_phase("preservation_preflight_completed")

        frame_times: list[float] = []
        try:
            try:
                with self._nvtx.range("trtvideo.initialization"):
                    self.log("\nInitializing TensorRT...")
                    self.runtime = self.create_runtime()
                    runtime = self.require_runtime()

                    if info.width != runtime.input_w or info.height != runtime.input_h:
                        print(
                            "ERROR: "
                            f"Video {info.width}x{info.height} does not match engine "
                            f"{runtime.input_w}x{runtime.input_h}",
                            file=sys.stderr,
                        )
                        raise SystemExit(1)
                    self._record_lifecycle_phase("runtime_initialized")

                    if args.profile or args.profile_json:
                        from trtvideo.diagnostics.profiling import ProfileCollector

                        self.profiler = ProfileCollector(
                            self.profile_stage_names(),
                            gpu_stages=self.gpu_stage_names(),
                            synchronize=runtime.synchronize,
                            skip_warmup=args.warmup_frames,
                        )

                    self.setup_decoder()
                    self._record_lifecycle_phase("decoder_initialized")
                    self.setup_encoder()
                    self._record_lifecycle_phase("encoder_initialized")

                    self.log(f"\nProcessing: {self.total_frames} frames")
                    self.log(f"Output: {args.output} ({runtime.output_w}x{runtime.output_h})\n")

                wall_start = time.perf_counter()
                with self._nvtx.range("trtvideo.frame_loop"):
                    self._run_loop(frame_times)
                self._record_lifecycle_phase("frame_loop_completed")
                with self._nvtx.range("trtvideo.finalize"):
                    self.finalize()
                self._record_lifecycle_phase("pipeline_finalized")
                wall_total = time.perf_counter() - wall_start
            finally:
                self.cleanup()
                self._record_lifecycle_phase("cleanup_completed")
            commit_atomic_output(working_output_path, args.output)
            self._record_lifecycle_phase("output_committed")
        except BaseException:
            working_output_path.unlink(missing_ok=True)
            raise
        finally:
            self._working_output_path = None

        self._record_lifecycle_phase("reporting_started")
        self._write_benchmark_lifecycle_markers(len(frame_times))

        # Profile table
        if args.profile and self.profiler and self.profiler.committed_count > 0:
            self.profiler.print_table(
                runtime.input_w,
                runtime.input_h,
                runtime.output_w,
                runtime.output_h,
                frame_times,
            )
        if args.profile_json and self.profiler:
            self._write_profile_json(args.profile_json, frame_times, wall_total)

        # Stats
        self._print_stats(frame_times, wall_total)

    def _record_frame_completed(self) -> None:
        """Capture benchmark lifecycle boundaries without enabling stage profiling."""
        if not getattr(self.args, "benchmark_lifecycle_json", None):
            return
        completed_ns = time.perf_counter_ns()
        if self._first_frame_completed_ns is None:
            self._first_frame_completed_ns = completed_ns
        self._last_frame_completed_ns = completed_ns

    def _record_lifecycle_phase(self, name: str) -> None:
        """Record an optional project-specific benchmark lifecycle checkpoint."""
        if not getattr(self.args, "benchmark_lifecycle_json", None):
            return
        self._lifecycle_phase_completed_ns[name] = time.perf_counter_ns()

    def _write_benchmark_lifecycle_markers(self, processed_frames: int) -> None:
        path = getattr(self.args, "benchmark_lifecycle_json", None)
        if path is None:
            return
        if (
            self._first_frame_completed_ns is None
            or self._last_frame_completed_ns is None
            or processed_frames <= 0
        ):
            return
        write_frame_markers(
            Path(path),
            FrameLifecycleMarkers(
                first_frame_completed_ns=self._first_frame_completed_ns,
                last_frame_completed_ns=self._last_frame_completed_ns,
                processed_frames=processed_frames,
                instrumentation="trtvideo-frame-loop",
                phase_completed_ns=dict(self._lifecycle_phase_completed_ns),
            ),
        )

    def _write_profile_json(
        self,
        output_path: str,
        frame_times: list[float],
        wall_total: float,
    ) -> None:
        runtime = self.require_runtime()
        profile = self.profiler.summary(frame_times) if self.profiler else {}
        stage_ms = profile.get("stage_ms", {})
        stage_key_map = self.profile_stage_key_map()
        normalized_stage_ms = {
            stage_key_map.get(name, name): value for name, value in stage_ms.items()
        }

        report = {
            "engine": self.engine_path,
            "gpu": runtime.gpu_name,
            "input": self.args.input,
            "output": self.args.output,
            "input_resolution": f"{self.info.width}x{self.info.height}",
            "output_resolution": f"{runtime.output_w}x{runtime.output_h}",
            "processed_frames": len(frame_times),
            "frames": profile.get("frames", len(frame_times)),
            "warmup_frames": profile.get("warmup_frames", 0),
            "processing_fps": profile.get("processing_fps", 0.0),
            "throughput_fps": len(frame_times) / wall_total if wall_total > 0 else 0.0,
            "avg_frame_sec": profile.get("avg_frame_sec", 0.0),
            "avg_frame_ms": profile.get("avg_frame_ms", 0.0),
            "min_frame_ms": profile.get("min_frame_ms", 0.0),
            "max_frame_ms": profile.get("max_frame_ms", 0.0),
            "wall_total_sec": wall_total,
            "stage_ms": normalized_stage_ms,
            "gpu_peak_mem_mb": runtime.peak_memory_allocated_mb(),
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _run_loop(self, frame_times: list[float]) -> None:
        """Frame loop. Subclasses can override for a tight loop."""
        args = self.args
        frame_idx = 0
        for raw_frame in iter_limited_frames(
            self.decode_frames(),
            limit=args.max_frames,
        ):
            t0 = time.perf_counter()
            self.process_frame(raw_frame)
            t1 = time.perf_counter()
            self.after_frame()
            self._record_frame_completed()

            frame_time = t1 - t0
            frame_times.append(frame_time)
            frame_idx += 1

            if not args.quiet and (frame_idx % args.log_interval == 0 or frame_idx == 1):
                recent = frame_times[-args.log_interval :]
                avg = sum(recent) / len(recent)
                fps = 1.0 / avg if avg > 0 else 0
                self.log(
                    f"  Frame {frame_idx}/{self.total_frames}  |  "
                    f"{frame_time:.3f}s  |  avg {avg:.3f}s/frame  |  "
                    f"{fps:.1f} fps"
                )

    def _print_stats(self, frame_times: list[float], wall_total: float) -> None:
        if not frame_times:
            return

        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        if len(frame_times) > 1:
            avg_no_warmup = sum(frame_times[1:]) / len(frame_times[1:])
            fps_no_warmup = 1.0 / avg_no_warmup if avg_no_warmup > 0 else 0
        else:
            avg_no_warmup = avg_time
            fps_no_warmup = fps

        wall_min = int(wall_total // 60)
        wall_sec = wall_total % 60

        print(f"\n{'='*50}")
        print(f"Frames processed: {len(frame_times)}")
        print(f"Average time:     {avg_time:.3f}s/frame ({fps:.1f} fps)")
        print(f"  Without warmup: {avg_no_warmup:.3f}s/frame ({fps_no_warmup:.1f} fps)")
        print(f"Min/Max:          {min_time:.3f}s / {max_time:.3f}s")
        print(f"Total time:       {wall_min}m {wall_sec:.1f}s")
        print(f"Output file:      {self.args.output}")
        print(f"{'='*50}")
