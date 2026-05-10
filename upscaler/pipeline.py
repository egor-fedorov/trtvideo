"""Base pipeline with template method pattern for the frame processing loop."""

import argparse
import os
import sys
import time
from abc import ABC, abstractmethod

import torch

from upscaler.engine import TRTInference
from upscaler.profiling import ProfileCollector
from upscaler.video import VideoInfo, get_video_info


class BasePipeline(ABC):
    """Base class for video upscaling pipeline.

    Template method: run() controls the full cycle.
    Subclasses implement hooks: add_extra_args, setup_decoder, setup_encoder,
    decode_frames, process_frame, finalize, cleanup.
    """

    DESCRIPTION: str = "TensorRT Video Upscaler"

    def __init__(self):
        parser = argparse.ArgumentParser(description=self.DESCRIPTION)
        parser.add_argument("--engine", required=True, help="Path to .engine file")
        parser.add_argument("--input", required=True, help="Input video")
        parser.add_argument("--output", default=None, help="Output video")
        parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
        parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
        parser.add_argument("--log-interval", type=int, default=10, help="Log every N frames")
        parser.add_argument(
            "--profile", action="store_true", help="Per-stage profiling (CUDA events)"
        )
        verbosity = parser.add_mutually_exclusive_group()
        verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
        verbosity.add_argument("--quiet", action="store_true", help="Minimal output")

        self.add_extra_args(parser)
        self.args = parser.parse_args()

        self.info = VideoInfo(width=0, height=0, fps=0.0, fps_str="0/1", nb_frames=0)
        self.trt_model: TRTInference | None = None
        self.total_frames: int = 0
        self.profiler: ProfileCollector | None = None

    # --- Logging helpers ---

    def log(self, *a, **kw):
        if not self.args.quiet:
            print(*a, **kw)

    def log_verbose(self, *a, **kw):
        if self.args.verbose:
            print(*a, **kw)

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

        if not os.path.exists(args.engine):
            print(f"ERROR: Engine not found: {args.engine}")
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

        self.log("\nInitializing TensorRT...")
        torch.cuda.set_device(args.gpu_id)
        self.trt_model = TRTInference(args.engine, quiet=args.quiet, gpu_id=args.gpu_id)

        if info.width != self.trt_model.input_w or info.height != self.trt_model.input_h:
            print(
                f"WARNING: Video {info.width}x{info.height} "
                f"!= engine {self.trt_model.input_w}x{self.trt_model.input_h}"
            )
            sys.exit(1)

        if args.profile:
            self.profiler = ProfileCollector(
                self.profile_stage_names(),
                gpu_stages=self.gpu_stage_names(),
            )

        self.setup_decoder()
        self.setup_encoder()

        self.log(f"\nProcessing: {self.total_frames} frames")
        self.log(
            f"Output: {args.output} " f"({self.trt_model.output_w}x{self.trt_model.output_h})\n"
        )

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
        if self.profiler and self.profiler.committed_count > 0:
            self.profiler.print_table(
                self.trt_model.input_w,
                self.trt_model.input_h,
                self.trt_model.output_w,
                self.trt_model.output_h,
                frame_times,
            )

        # Stats
        self._print_stats(frame_times, wall_total)

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
