"""Memory layout checks for static, linear TensorRT bindings."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

from trtvideo.models.manifest import TensorSpec


class TensorBindingError(RuntimeError):
    """A GPU buffer cannot be used with the engine's I/O contract."""


def validate_binding_strides(
    spec: TensorSpec,
    *,
    trt_element_strides: Sequence[int],
    cuda_byte_strides: Sequence[int] | None,
) -> None:
    """Compare byte offsets for all reachable indices of a linear binding."""
    itemsize = {"fp16": 2, "fp32": 4}[spec.dtype]
    expected = tuple(stride * itemsize for stride in trt_element_strides)
    shape = spec.shape
    # An omitted CUDA Array Interface stride tuple denotes C-contiguous memory.
    actual = (
        tuple(prod(shape[i + 1 :]) * itemsize for i in range(len(shape)))
        if cuda_byte_strides is None
        else tuple(cuda_byte_strides)
    )
    if not (len(actual) == len(expected) == len(shape)) or any(
        size > 1 and got != want for size, got, want in zip(shape, actual, expected, strict=True)
    ):
        raise TensorBindingError(
            f"Incompatible binding {spec.name!r} (shape={shape}, dtype={spec.dtype}): "
            f"TensorRT expects byte strides {expected}, CV-CUDA provides {actual}"
        )
