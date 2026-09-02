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
    result_json_path: Path | None = None
    progress_jsonl_path: Path | None = None
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
        report_destinations = {
            "--profile-json": self.profile_json_path,
            "--result-json": self.result_json_path,
            "--progress-jsonl": self.progress_jsonl_path,
            "--benchmark-lifecycle-json": self.benchmark_lifecycle_path,
        }
        selected = [(name, path) for name, path in report_destinations.items() if path is not None]
        for index, (left_name, left_path) in enumerate(selected):
            for right_name, right_path in selected[index + 1 :]:
                if _destination_key(left_path) == _destination_key(right_path):
                    raise PipelineError(
                        f"{left_name} and {right_name} must use different destinations"
                    )
        protected_paths = {
            _destination_key(self.engine_path),
            _destination_key(self.input_path),
            _destination_key(self.output_path),
        }
        for name, destination in report_destinations.items():
            if destination is not None and _destination_key(destination) in protected_paths:
                raise PipelineError(f"{name} must not overwrite an engine, input, or video output")


def default_output_path(input_path: Path) -> Path:
    """Return the implicit output path used when --output is omitted."""
    return input_path.with_name(f"{input_path.stem}_processed{input_path.suffix}")


def _destination_key(path: Path) -> str:
    if path == Path("-"):
        return "-"
    return str(path.expanduser().resolve())
