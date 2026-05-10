"""Base pipeline with template method pattern for the frame processing loop."""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod

import torch

from upscaler.engine import TensorRTRuntime
from upscaler.engine_registry import (
    format_registry_entries,
    load_engine_registry,
    select_engine_for_video,
)
from upscaler.profiling import ProfileCollector
from upscaler.runtime import RuntimeEngine
from upscaler.video import VideoInfo, get_video_info


class BasePipeline(ABC):
    """Base class for video upscaling pipeline.

    Template method: run() controls the full cycle.
    Subclasses implement hooks: add_extra_args, setup_decoder, setup_encoder,
    decode_frames, process_frame, finalize, cleanup.
    """

    DESCRIPTION: str = "TensorRT Video Upscaler"
    BACKEND_NAME: str = "unknown"

    def __init__(self):
        parser = argparse.ArgumentParser(description=self.DESCRIPTION)
        engine_source = parser.add_mutually_exclusive_group(required=True)
        engine_source.add_argument("--engine", help="Path to .engine file")
        engine_source.add_argument(
            "--model",
            help="Path to model registry directory, registry manifest, or engine manifest",
        )
        parser.add_argument("--input", required=True, help="Input video")
        parser.add_argument("--output", default=None, help="Output video")
        parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
        parser.add_argument(
            "--engine-precision",
            choices=["fp16", "fp32"],
            default=None,
            help="Preferred precision when selecting an engine from --model",
        )
        parser.add_argument(
            "--engine-io-precision",
            choices=["fp16", "fp32"],
            default=None,
            help="Preferred input/output binding precision when selecting an engine from --model",
        )
        parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
        parser.add_argument(
            "--warmup-frames",
            type=int,
            default=1,
            help="Frames to exclude from profiling/benchmark summaries",
        )
        parser.add_argument("--log-interval", type=int, default=10, help="Log every N frames")
        parser.add_argument(
            "--profile", action="store_true", help="Per-stage profiling (CUDA events)"
        )
        parser.add_argument("--profile-json", default=None, help="Write profiling JSON summary")
        verbosity = parser.add_mutually_exclusive_group()
        verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
        verbosity.add_argument("--quiet", action="store_true", help="Minimal output")

        self.add_extra_args(parser)
        self.args = parser.parse_args()

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
    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments specific to the subclass."""

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
        self.total_frames = args.max_frames if args.max_frames > 0 else info.nb_frames
        self.engine_path = self.resolve_engine_path(info)

        self.log("\nInitializing TensorRT...")
        torch.cuda.set_device(args.gpu_id)
        self.runtime = TensorRTRuntime(self.engine_path, quiet=args.quiet, gpu_id=args.gpu_id)
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
