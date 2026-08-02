"""Typed configuration and errors for the production video pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class PipelineError(RuntimeError):
    """Raised when the requested production pipeline cannot run."""


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """Validated input to the GPU-resident processing pipeline."""

    engine_path: Path
    input_path: Path
    output_path: Path
    gpu_id: int = 0
    max_frames: int = 0
    warmup_frames: int = 1
    log_interval: int = 10
    profile: bool = False
    profile_json_path: Path | None = None
    benchmark_lifecycle_path: Path | None = None
    bitrate_mbps: float | None = None
    codec: Literal["h264", "hevc"] = "h264"
    verbose: bool = False
    quiet: bool = False

    def __post_init__(self) -> None:
        if self.gpu_id < 0:
            raise PipelineError("--gpu-id must be non-negative")
        if self.max_frames < 0:
            raise PipelineError("--max-frames must be non-negative")
        if self.warmup_frames < 0:
            raise PipelineError("--warmup-frames must be non-negative")
        if self.log_interval <= 0:
            raise PipelineError("--log-interval must be greater than zero")
        if self.bitrate_mbps is not None and self.bitrate_mbps <= 0:
            raise PipelineError("--bitrate-mbps must be greater than zero")


def default_output_path(input_path: Path) -> Path:
    """Return the implicit output path used when --output is omitted."""
    return input_path.with_name(f"{input_path.stem}_processed{input_path.suffix}")
