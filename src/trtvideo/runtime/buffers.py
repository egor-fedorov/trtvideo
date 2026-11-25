"""Structural types for zero-copy CUDA buffer interchange, without GPU imports."""

from typing import NotRequired, Protocol, TypedDict


class CudaArrayInterface(TypedDict):
    version: int
    shape: tuple[int, ...]
    typestr: str
    data: tuple[int, bool]
    strides: NotRequired[tuple[int, ...] | None]
    stream: NotRequired[int | None]


class CudaArray(Protocol):
    @property
    def __cuda_array_interface__(self) -> CudaArrayInterface: ...


class CudaBufferProvider(Protocol):
    def cuda(self) -> CudaArray: ...


class DLPackProvider(Protocol):
    def __dlpack__(self, stream: int | None = None) -> object: ...
