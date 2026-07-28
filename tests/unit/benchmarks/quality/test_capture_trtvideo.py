from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np

from benchmarks.scripts.quality.capture_trtvideo import _copy_gpu_tensor_to_host


class FakeCudaTensor:
    def __init__(self, array: np.ndarray):
        self._array = array

    def cuda(self) -> SimpleNamespace:
        return SimpleNamespace(
            __cuda_array_interface__={
                "version": 3,
                "shape": self._array.shape,
                "typestr": self._array.dtype.str,
                "data": (int(self._array.ctypes.data), False),
                "strides": self._array.strides,
            }
        )


class FakeCudaRuntime:
    class cudaError_t:
        cudaSuccess = 0

    class cudaMemcpyKind:
        cudaMemcpyDeviceToHost = 2

    @staticmethod
    def cudaMemcpy2D(
        destination: int,
        destination_pitch: int,
        source: int,
        source_pitch: int,
        width: int,
        height: int,
        _kind: int,
    ) -> tuple[int]:
        for row in range(height):
            ctypes.memmove(
                destination + row * destination_pitch,
                source + row * source_pitch,
                width,
            )
        return (FakeCudaRuntime.cudaError_t.cudaSuccess,)


def test_copy_gpu_tensor_to_host_preserves_nchw_values() -> None:
    source = np.arange(12, dtype=np.float16).reshape(1, 3, 2, 2)

    copied = _copy_gpu_tensor_to_host(
        FakeCudaTensor(source),
        cudart=FakeCudaRuntime,
        np=np,
    )

    np.testing.assert_array_equal(copied, source)
