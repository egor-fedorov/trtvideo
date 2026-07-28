"""Runtime engine protocols used by video pipelines."""

from typing import Any, Protocol

from trtvideo.models.manifest import ModelSpec, TensorSpec

TensorLike = Any
CudaStream = Any


class RuntimeEngine(Protocol):
    """Runtime contract used by the video pipeline."""

    model_spec: ModelSpec
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    input_name: str
    output_name: str
    input_dtype: Any
    output_dtype: Any
    stream: CudaStream
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
