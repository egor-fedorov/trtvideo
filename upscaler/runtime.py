"""Runtime engine protocol used by video pipelines."""

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from upscaler.model_spec import ModelSpec, TensorSpec

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
    input_w: int
    input_h: int
    output_w: int
    output_h: int

    def infer(
        self,
        inputs: Mapping[str, TensorLike],
        *,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> dict[str, TensorLike]:
        """Run raw runtime inference on prepared tensors."""

    def infer_rgb_cpu(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Run full CPU RGB frame -> runtime -> CPU RGB frame inference."""

    def infer_rgb_cpu_profiled(
        self,
        frame_rgb: np.ndarray,
        events: tuple[CudaStream, CudaStream, CudaStream, CudaStream],
    ) -> np.ndarray:
        """Run CPU RGB inference and record preprocess/runtime/postprocess CUDA events."""

    def infer_rgb_tensor(
        self,
        rgb_hwc: TensorLike,
        *,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> TensorLike:
        """Run GPU RGB HWC tensor -> runtime -> GPU RGB HWC tensor inference."""

    def infer_rgb_tensor_into(
        self,
        rgb_hwc: TensorLike,
        output_rgb_hwc: TensorLike,
        *,
        input_nchw: TensorLike | None = None,
        output_rgb_float: TensorLike | None = None,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> TensorLike:
        """Run GPU RGB HWC tensor inference into preallocated output buffers."""
