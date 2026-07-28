"""Shared argv builder for VapourSynth-to-NVENC benchmark pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.scripts.runtime.command import CommandSpec, command_spec
from trtvideo.video.nvcodec.encoder import NvencCbrContract


@dataclass(frozen=True)
class VspipeNvencConfig:
    """Immutable inputs shared by the vstrt and pinned VSGAN command paths."""

    script: str
    source: str
    engine: str
    gpu_id: int
    requests: int | None
    cuda_graph: bool
    num_streams: int
    encoder: NvencCbrContract
    script_arguments: tuple[tuple[str, str], ...] = ()

    def build(self, *, output_path: Path, frames: int) -> CommandSpec:
        if frames <= 0:
            raise ValueError("frames must be positive")
        vspipe = [
            "vspipe",
            "--container",
            "y4m",
            "--progress",
            "--start",
            "0",
            "--end",
            str(frames - 1),
        ]
        if self.requests is not None:
            vspipe.extend(["--requests", str(self.requests)])
        vspipe.extend(
            [
                "--arg",
                f"source={self.source}",
                "--arg",
                f"engine={self.engine}",
                "--arg",
                f"gpu_id={self.gpu_id}",
                "--arg",
                f"cuda_graph={int(self.cuda_graph)}",
                "--arg",
                f"num_streams={self.num_streams}",
            ]
        )
        for name, value in self.script_arguments:
            vspipe.extend(["--arg", f"{name}={value}"])
        vspipe.extend([self.script, "-"])

        ffmpeg = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "yuv4mpegpipe",
            "-i",
            "pipe:0",
            "-frames:v",
            str(frames),
            "-an",
            "-sn",
            "-dn",
            *self.encoder.ffmpeg_options(),
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        return command_spec(vspipe, ffmpeg)
