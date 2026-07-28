"""Small CUDA Runtime API adapters used without importing PyTorch."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import cuda.bindings.runtime as cudart


def _checked(result: Any, operation: str) -> tuple[Any, ...]:
    values = result if isinstance(result, tuple) else (result,)
    error = values[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")
    return tuple(values[1:])


def set_device(device_id: int) -> None:
    """Select and initialize the CUDA device for the current host thread."""
    _checked(cudart.cudaSetDevice(device_id), "cudaSetDevice")


def device_name(device_id: int) -> str:
    """Return the CUDA device name, falling back to its ordinal."""
    (properties,) = _checked(
        cudart.cudaGetDeviceProperties(device_id),
        "cudaGetDeviceProperties",
    )
    name = properties.name
    if isinstance(name, bytes):
        return name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(name)


def _stream_handle(stream: Any) -> int:
    if isinstance(stream, int):
        return stream
    if hasattr(stream, "handle"):
        return int(stream.handle)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    raise TypeError(f"Unsupported CUDA stream object: {type(stream)!r}")


class CudaEvent:
    """CUDA timing event with the subset used by ProfileCollector."""

    def __init__(self) -> None:
        (self._event,) = _checked(cudart.cudaEventCreate(), "cudaEventCreate")
        self._closed = False

    def record(self, stream: Any) -> None:
        """Record this event on the supplied CUDA stream."""
        _checked(
            cudart.cudaEventRecord(self._event, _stream_handle(stream)),
            "cudaEventRecord",
        )

    def elapsed_time(self, end_event: CudaEvent) -> float:
        """Return elapsed milliseconds from this event to end_event."""
        (milliseconds,) = _checked(
            cudart.cudaEventElapsedTime(self._event, end_event._event),
            "cudaEventElapsedTime",
        )
        return float(milliseconds)

    def close(self) -> None:
        """Release the CUDA event."""
        if self._closed:
            return
        _checked(cudart.cudaEventDestroy(self._event), "cudaEventDestroy")
        self._closed = True

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        with suppress(Exception):
            self.close()
