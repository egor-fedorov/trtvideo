"""Base pipeline with template method pattern for the frame processing loop."""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod

import torch

from upscaler.models.registry import (
    format_registry_entries,
    load_engine_registry,
    select_engine_for_video,
)
from upscaler.profiling import ProfileCollector
from upscaler.runtime import RuntimeEngine
from upscaler.runtime.tensorrt import TensorRTRuntime
from upscaler.video.info import VideoInfo, get_video_info

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
    BACKEND_NAME: str = "unknown"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.info = VideoInfo(width=0, height=0, fps=0.0, fps_str="0/1", nb_frames=0)
        self.runtime: RuntimeEngine | None = None
        self.engine_path: str = ""
        self.total_frames: int = 0
        self.profiler: ProfileCollector | None = None

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

    def ffmpeg_limited_duration_args(self) -> list[str]:
        """Return ffmpeg output args that keep audio aligned with --max-frames."""
        if self.args.max_frames <= 0 or self.info.fps <= 0:
            return []
        duration_sec = self.args.max_frames / self.info.fps
        return ["-t", f"{duration_sec:.6f}", "-shortest"]

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

    def resolve_engine_path(self, video_info: VideoInfo) -> str:
        """Resolve explicit engine path or select one from a model registry."""
        if self.args.engine:
            return self.args.engine

        try:
            entries = load_engine_registry(self.args.model)
        except (OSError, ValueError) as exc:
            print(f"ERROR: Failed to load model registry: {exc}")
            sys.exit(1)
        selected = select_engine_for_video(
            entries,
            video_info,
            precision=self.args.engine_precision,
            io_precision=self.args.engine_io_precision,
        )
        if selected is None:
            print(
                "ERROR: No compatible static engine found in model registry "
                f"for {video_info.width}x{video_info.height}"
            )
            if self.args.engine_precision:
                print(f"  Requested precision: {self.args.engine_precision}")
            if self.args.engine_io_precision:
                print(f"  Requested I/O precision: {self.args.engine_io_precision}")
            print("Available engines:")
            print(format_registry_entries(entries))
            sys.exit(1)

        self.log(f"Selected engine: {selected.engine_path}")
        if selected.manifest_path:
            self.log_verbose(f"Engine manifest: {selected.manifest_path}")
        return selected.engine_path

    # --- Abstract hooks ---

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
        if args.model and not os.path.exists(args.model):
            print(f"ERROR: Model registry not found: {args.model}")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"ERROR: Video not found: {args.input}")
            sys.exit(1)

        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_upscaled{ext}"

        self.info = get_video_info(args.input)
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
        self.engine_path = self.resolve_engine_path(info)

        self.log("\nInitializing TensorRT...")
        torch.cuda.set_device(args.gpu_id)
        self.runtime = TensorRTRuntime(
            self.engine_path,
            quiet=args.quiet,
            gpu_id=args.gpu_id,
            use_cuda_graph=args.cuda_graph,
        )
        runtime = self.require_runtime()

        if info.width != runtime.input_w or info.height != runtime.input_h:
            print(
                f"WARNING: Video {info.width}x{info.height} "
                f"!= engine {runtime.input_w}x{runtime.input_h}"
            )
            sys.exit(1)

        if args.profile or args.profile_json:
            self.profiler = ProfileCollector(
                self.profile_stage_names(),
                gpu_stages=self.gpu_stage_names(),
                skip_warmup=args.warmup_frames,
            )

        self.setup_decoder()
        self.setup_encoder()

        self.log(f"\nProcessing: {self.total_frames} frames")
        self.log(f"Output: {args.output} ({runtime.output_w}x{runtime.output_h})\n")

        frame_times: list[float] = []
        wall_start = time.perf_counter()

        try:
            self._run_loop(frame_times)
            self.finalize()
        except BrokenPipeError:
            print("WARNING: Encoder pipe closed")
        finally:
            self.cleanup()

        wall_total = time.perf_counter() - wall_start

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

        try:
            gpu_name = torch.cuda.get_device_name(self.args.gpu_id)
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated(self.args.gpu_id) / (1024 * 1024)
        except RuntimeError:
            gpu_name = f"cuda:{self.args.gpu_id}"
            gpu_peak_mem_mb = 0.0

        report = {
            "backend": self.BACKEND_NAME,
            "model": self.args.model,
            "engine": self.engine_path,
            "gpu": gpu_name,
            "input": self.args.input,
            "output": self.args.output,
            "input_resolution": f"{self.info.width}x{self.info.height}",
            "output_resolution": f"{runtime.output_w}x{runtime.output_h}",
            "cuda_graph_requested": self.args.cuda_graph,
            "cuda_graph": runtime.cuda_graph_enabled,
            "cuda_graph_error": runtime.cuda_graph_error,
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
            "gpu_peak_mem_mb": gpu_peak_mem_mb,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    def _run_loop(self, frame_times: list[float]) -> None:
        """Frame loop. Subclasses can override for a tight loop."""
        args = self.args
        frame_idx = 0
        for raw_frame in self.decode_frames():
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            t0 = time.perf_counter()
            self.process_frame(raw_frame)
            t1 = time.perf_counter()
            self.after_frame()

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
