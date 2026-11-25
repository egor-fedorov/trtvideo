from __future__ import annotations

import ctypes
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.scripts.quality.capture_trtvideo import _copy_gpu_tensor_to_host
from benchmarks.scripts.quality.model_space import ModelSpaceError


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


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.bindings = bindings
    bindings.runtime = FakeCudaRuntime
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", FakeCudaRuntime)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("padded", [False, True])
def test_copy_gpu_tensor_to_host_preserves_nchw_values(dtype, padded) -> None:
    source = np.arange(24, dtype=dtype).reshape(1, 3, 2, 4)
    if padded:
        source = source[:, :, :, :2]

    copied = _copy_gpu_tensor_to_host(FakeCudaTensor(source))

    np.testing.assert_array_equal(copied, source)
    assert copied.dtype == source.dtype
    assert copied.flags.c_contiguous


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (np.zeros((3, 2, 2), dtype=np.float32), "shape"),
        (np.zeros((1, 3, 2, 2), dtype=np.uint8), "dtype"),
        (np.zeros((1, 3, 2, 4), dtype=np.float32)[:, :, :, ::2], "not contiguous"),
    ],
)
def test_capture_rejects_unsupported_buffers(source, error) -> None:
    with pytest.raises(ModelSpaceError, match=error):
        _copy_gpu_tensor_to_host(FakeCudaTensor(source))


def test_capture_reports_cuda_copy_failure(monkeypatch) -> None:
    monkeypatch.setattr(FakeCudaRuntime, "cudaMemcpy2D", lambda *args: (1,))
    with pytest.raises(ModelSpaceError, match="Failed to copy captured tensor to host"):
        _copy_gpu_tensor_to_host(FakeCudaTensor(np.zeros((1, 3, 2, 2), dtype=np.float32)))
