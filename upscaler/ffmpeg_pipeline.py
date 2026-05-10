"""FFmpeg pipe-based video upscaling pipeline."""

import argparse
import subprocess
import time

import numpy as np
import torch

from upscaler.pipeline import BasePipeline


_GPU_STAGES = [
    "Preprocess (CPU\u2192GPU)",
    "TRT inference",
    "Postprocess (GPU\u2192CPU)",
]


class FfmpegPipeline(BasePipeline):
    """Pipeline: ffmpeg decode (pipe) \u2192 TRT \u2192 ffmpeg encode (pipe)."""

    DESCRIPTION = "TensorRT Video Upscaler (ffmpeg backend)"

    def __init__(self):
        self._decoder: subprocess.Popen | None = None
        self._encoder: subprocess.Popen | None = None
        self._frame_size_in: int = 0
        self._frame_size_out: int = 0
        super().__init__()

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--crf", type=int, default=18, help="CRF for x264")

    def profile_stage_names(self) -> list[str]:
        return [
            "ffmpeg decode (pipe read)",
            *_GPU_STAGES,
            "ffmpeg encode (pipe write)",
        ]

    def gpu_stage_names(self) -> list[str]:
        return _GPU_STAGES

    def setup_decoder(self) -> None:
        self._frame_size_in = self.trt_model.input_w * self.trt_model.input_h * 3
        decode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.args.input,
        ]
        if self.args.max_frames > 0:
            decode_cmd += ["-frames:v", str(self.args.max_frames)]
        decode_cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        self.log_verbose(f"Decode cmd: {' '.join(decode_cmd)}")
        self._decoder = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
            bufsize=self._frame_size_in * 4,
        )

    def setup_encoder(self) -> None:
        self._frame_size_out = self.trt_model.output_w * self.trt_model.output_h * 3
        encode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.trt_model.output_w}x{self.trt_model.output_h}",
            "-r",
            self.info["fps_str"],
            "-i",
            "pipe:0",
            "-i",
            self.args.input,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(self.args.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            self.args.output,
        ]
        self.log_verbose(f"Encode cmd: {' '.join(encode_cmd)}")
        self._encoder = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            bufsize=self._frame_size_out * 4,
        )

    def _run_loop(self, frame_times: list[float]) -> None:
        if self.profiler:
            self._run_loop_profiled(frame_times)
            return

        # Tight loop
        pipe_read = self._decoder.stdout.read
        pipe_write = self._encoder.stdin.write
        infer = self.trt_model.infer
        frame_size = self._frame_size_in
        in_h, in_w = self.trt_model.input_h, self.trt_model.input_w
        max_frames = self.args.max_frames
        quiet = self.args.quiet
        log_interval = self.args.log_interval
        frame_idx = 0

        while True:
            raw = pipe_read(frame_size)
            if len(raw) < frame_size:
                break

            frame_rgb = np.frombuffer(raw, dtype=np.uint8).copy().reshape(in_h, in_w, 3)

            t0 = time.perf_counter()
            upscaled = infer(frame_rgb)
            t1 = time.perf_counter()

            pipe_write(upscaled.tobytes())

            frame_time = t1 - t0
            frame_times.append(frame_time)
            frame_idx += 1

            if not quiet and (frame_idx % log_interval == 0 or frame_idx == 1):
                recent = frame_times[-log_interval:]
                avg = sum(recent) / len(recent)
                fps = 1.0 / avg if avg > 0 else 0
                self.log(
                    f"  Frame {frame_idx}/{self.total_frames}  |  "
                    f"{frame_time:.3f}s  |  avg {avg:.3f}s/frame  |  "
                    f"{fps:.1f} fps"
                )

            if max_frames > 0 and frame_idx >= max_frames:
                break

    def _run_loop_profiled(self, frame_times: list[float]) -> None:
        """Profiled loop with CUDA events and wall-clock pipe I/O measurements."""
        pipe_read = self._decoder.stdout.read
        pipe_write = self._encoder.stdin.write
        frame_size = self._frame_size_in
        in_h, in_w = self.trt_model.input_h, self.trt_model.input_w
        max_frames = self.args.max_frames
        quiet = self.args.quiet
        log_interval = self.args.log_interval
        profiler = self.profiler
        trt_model = self.trt_model
        frame_idx = 0

        while True:
            tp0 = time.perf_counter()
            raw = pipe_read(frame_size)
            if len(raw) < frame_size:
                break
            frame_rgb = np.frombuffer(raw, dtype=np.uint8).copy().reshape(in_h, in_w, 3)
            profiler.record_wall_time("ffmpeg decode (pipe read)", time.perf_counter() - tp0)

            t0 = time.perf_counter()

            e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
            stream = trt_model.stream
            with torch.cuda.stream(stream):
                e0.record(stream)
                frame_gpu = torch.from_numpy(frame_rgb).to(
                    device=trt_model.device,
                    non_blocking=True,
                )
                frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                trt_model.gpu_input.copy_(frame_gpu)
                e1.record(stream)
                trt_model.context.execute_async_v3(stream_handle=stream.cuda_stream)
                e2.record(stream)
                output = trt_model.gpu_output.squeeze(0).permute(1, 2, 0)
                output = output.mul_(255.0).clamp_(0, 255).byte()
                e3.record(stream)
            stream.synchronize()
            upscaled = output.cpu().numpy()

            t1 = time.perf_counter()

            tw0 = time.perf_counter()
            pipe_write(upscaled.tobytes())
            profiler.record_wall_time("ffmpeg encode (pipe write)", time.perf_counter() - tw0)
            profiler.commit((e0, e1, e2, e3))

            frame_time = t1 - t0
            frame_times.append(frame_time)
            frame_idx += 1

            if not quiet and (frame_idx % log_interval == 0 or frame_idx == 1):
                recent = frame_times[-log_interval:]
                avg = sum(recent) / len(recent)
                fps = 1.0 / avg if avg > 0 else 0
                self.log(
                    f"  Frame {frame_idx}/{self.total_frames}  |  "
                    f"{frame_time:.3f}s  |  avg {avg:.3f}s/frame  |  "
                    f"{fps:.1f} fps"
                )

            if max_frames > 0 and frame_idx >= max_frames:
                break

    def decode_frames(self):
        raise NotImplementedError  # logic in _run_loop

    def process_frame(self, raw_frame) -> None:
        raise NotImplementedError  # logic in _run_loop

    def finalize(self) -> None:
        pass  # ffmpeg flushes on stdin close

    def cleanup(self) -> None:
        if self._decoder and self._decoder.stdout:
            self._decoder.stdout.close()
            self._decoder.wait()
        if self._encoder:
            if self._encoder.stdin:
                self._encoder.stdin.close()
            self._encoder.wait()
