"""Runtime engine protocols used by video pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from trtvideo.models.manifest import ModelSpec, TensorSpec

if TYPE_CHECKING:
    import cvcuda


class RuntimeEngine(Protocol):
    """Runtime contract used by the video pipeline."""

    model_spec: ModelSpec
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    input_name: str
    output_name: str
    input_dtype: cvcuda.Type
    output_dtype: cvcuda.Type
    stream: cvcuda.Stream
    stream_handle: int
    gpu_name: str
    input_w: int
    input_h: int
    output_w: int
    output_h: int

    def synchronize(self) -> None:
        """Wait for work queued on the runtime stream."""

    def peak_memory_allocated_mb(self) -> float:
        """Return runtime-tracked peak device memory when available."""
