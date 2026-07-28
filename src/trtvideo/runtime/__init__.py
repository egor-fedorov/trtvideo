"""Runtime engine protocols used by video pipelines."""

from typing import Any, Protocol

from trtvideo.models.manifest import ModelSpec, TensorSpec

TensorLike = Any
CudaStream = Any


class RuntimeEngine(Protocol):
    """Common runtime contract for current and future inference backends."""

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
    cuda_graph_enabled: bool
    cuda_graph_error: str | None
    input_w: int
    input_h: int
    output_w: int
    output_h: int

    def synchronize(self) -> None:
        """Wait for work queued on the runtime stream."""

    def peak_memory_allocated_mb(self) -> float:
        """Return runtime-tracked peak device memory when available."""


class CpuRgbRuntime(RuntimeEngine, Protocol):
    """Runtime contract required by the FFmpeg CPU-frame pipeline."""

    def infer_rgb_cpu(self, frame_rgb: Any) -> Any:
        """Run full CPU RGB frame -> runtime -> CPU RGB frame inference."""

    def infer_rgb_cpu_profiled(
        self,
        frame_rgb: Any,
        events: tuple[CudaStream, CudaStream, CudaStream, CudaStream],
    ) -> Any:
        """Run CPU RGB inference and record preprocess/runtime/postprocess CUDA events."""
