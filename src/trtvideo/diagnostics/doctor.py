"""Static runtime readiness checks for the Docker-first production path."""

from __future__ import annotations

import os
from ctypes import CDLL
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import disk_usage, which
from subprocess import run
from typing import Any

_GIB = 1024**3


@dataclass(frozen=True)
class CheckResult:
    """One independently reported environment check."""

    component: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    """Complete static environment readiness report."""

    checks: tuple[CheckResult, ...]

    @property
    def ready(self) -> bool:
        """Return whether every required component passed."""
        return all(check.passed for check in self.checks)


@dataclass(frozen=True)
class _CudaSnapshot:
    name: str
    runtime_version: int
    driver_api_version: int
    total_memory: int
    free_memory: int


def _error_detail(exc: Exception) -> str:
    detail = " ".join(str(exc).split())
    return detail or type(exc).__name__


def _failed(component: str, exc: Exception) -> CheckResult:
    return CheckResult(component=component, passed=False, detail=_error_detail(exc))


def _cuda_checked(cudart: Any, result: Any, operation: str) -> tuple[Any, ...]:
    values = result if isinstance(result, tuple) else (result,)
    if not values:
        raise RuntimeError(f"{operation} returned no status")
    if values[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {values[0]}")
    return tuple(values[1:])


def _cuda_version(value: int) -> str:
    major = value // 1000
    minor = (value % 1000) // 10
    patch = value % 10
    return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"


def _device_name(properties: Any) -> str:
    name = properties.name
    if isinstance(name, bytes):
        return name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(name)


def _inspect_cuda(gpu_id: int, *, importer: Any = import_module) -> _CudaSnapshot:
    cudart = importer("cuda.bindings.runtime")
    (device_count,) = _cuda_checked(cudart, cudart.cudaGetDeviceCount(), "cudaGetDeviceCount")
    if gpu_id < 0 or gpu_id >= int(device_count):
        raise RuntimeError(f"GPU {gpu_id} is unavailable; CUDA reports {device_count} device(s)")

    _cuda_checked(cudart, cudart.cudaSetDevice(gpu_id), "cudaSetDevice")
    (properties,) = _cuda_checked(
        cudart,
        cudart.cudaGetDeviceProperties(gpu_id),
        "cudaGetDeviceProperties",
    )
    (runtime_version,) = _cuda_checked(
        cudart,
        cudart.cudaRuntimeGetVersion(),
        "cudaRuntimeGetVersion",
    )
    (driver_api_version,) = _cuda_checked(
        cudart,
        cudart.cudaDriverGetVersion(),
        "cudaDriverGetVersion",
    )
    free_memory, total_memory = _cuda_checked(cudart, cudart.cudaMemGetInfo(), "cudaMemGetInfo")
    return _CudaSnapshot(
        name=_device_name(properties),
        runtime_version=int(runtime_version),
        driver_api_version=int(driver_api_version),
        total_memory=int(total_memory),
        free_memory=int(free_memory),
    )


def check_docker(
    *,
    marker: Path = Path("/.dockerenv"),
    command_runner: Any = run,
) -> CheckResult:
    """Verify Docker execution, or a reachable daemon for a direct host install."""
    if marker.exists():
        return CheckResult("Docker", True, "container detected")

    executable = which("docker")
    if executable is None:
        return CheckResult("Docker", False, "not running in Docker and docker is unavailable")
    try:
        completed = command_runner(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return _failed("Docker", exc)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "docker daemon is unreachable"
        return CheckResult("Docker", False, detail)
    return CheckResult("Docker", True, f"daemon {completed.stdout.strip() or 'reachable'}")


def check_driver(gpu_id: int, *, command_runner: Any = run) -> CheckResult:
    """Read the host NVIDIA driver version mounted into the container."""
    try:
        completed = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(gpu_id),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return _failed("Driver", exc)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "nvidia-smi failed"
        return CheckResult("Driver", False, detail)
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(versions) != 1:
        return CheckResult("Driver", False, "nvidia-smi returned no unambiguous driver version")
    return CheckResult("Driver", True, versions[0])


def check_cuda(gpu_id: int, *, importer: Any = import_module) -> tuple[CheckResult, ...]:
    """Initialize CUDA and report GPU identity, API compatibility, and VRAM."""
    try:
        snapshot = _inspect_cuda(gpu_id, importer=importer)
    except Exception as exc:
        detail = _error_detail(exc)
        return tuple(CheckResult(component, False, detail) for component in ("GPU", "CUDA", "VRAM"))

    return (
        CheckResult("GPU", True, f"device {gpu_id}: {snapshot.name}"),
        CheckResult(
            "CUDA",
            True,
            (
                f"runtime {_cuda_version(snapshot.runtime_version)}, "
                f"driver API {_cuda_version(snapshot.driver_api_version)}"
            ),
        ),
        CheckResult(
            "VRAM",
            snapshot.total_memory > 0 and snapshot.free_memory > 0,
            (
                f"{snapshot.free_memory / _GIB:.2f} GiB free of "
                f"{snapshot.total_memory / _GIB:.2f} GiB"
            ),
        ),
    )


def check_tensorrt(*, importer: Any = import_module) -> CheckResult:
    """Create a TensorRT runtime without loading a model engine."""
    try:
        tensorrt = importer("tensorrt")
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        runtime = tensorrt.Runtime(logger)
        if runtime is None:
            raise RuntimeError("TensorRT runtime creation returned null")
    except Exception as exc:
        return _failed("TensorRT", exc)
    version = str(getattr(tensorrt, "__version__", "unknown version"))
    return CheckResult("TensorRT", True, version)


def check_cvcuda(*, importer: Any = import_module) -> CheckResult:
    """Create a minimal CV-CUDA stream and device tensor."""
    try:
        cvcuda = importer("cvcuda")
        stream = cvcuda.Stream()
        tensor = cvcuda.Tensor((1, 1, 1, 3), cvcuda.Type.U8, layout="NHWC")
        if stream is None or tensor is None:
            raise RuntimeError("CV-CUDA initialization returned null")
    except Exception as exc:
        return _failed("CV-CUDA", exc)
    version = str(getattr(cvcuda, "__version__", "version unavailable"))
    return CheckResult("CV-CUDA", True, f"{version}; device allocation succeeded")


def check_driver_library(
    component: str,
    soname: str,
    symbol: str,
    *,
    loader: Any = CDLL,
) -> CheckResult:
    """Load an NVIDIA video driver library and resolve its public entry point."""
    try:
        library = loader(soname)
        getattr(library, symbol)
    except (AttributeError, OSError) as exc:
        return _failed(component, exc)
    return CheckResult(component, True, f"{soname}: {symbol} available")


def check_pynvvideocodec(*, importer: Any = import_module) -> CheckResult:
    """Verify the production Python NVDEC/NVENC binding surface."""
    try:
        module = importer("PyNvVideoCodec")
        missing = [
            name for name in ("ThreadedDecoder", "CreateEncoder") if not hasattr(module, name)
        ]
        if missing:
            raise RuntimeError(f"missing API: {', '.join(missing)}")
    except Exception as exc:
        return _failed("PyNvVideoCodec", exc)
    version = str(getattr(module, "__version__", "version unavailable"))
    return CheckResult("PyNvVideoCodec", True, version)


def check_disk(path: Path) -> CheckResult:
    """Report usable disk capacity for the selected output filesystem."""
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"not a directory: {resolved}")
        if not os.access(resolved, os.W_OK):
            raise RuntimeError(f"directory is not writable: {resolved}")
        usage = disk_usage(resolved)
        if usage.free <= 0:
            raise RuntimeError(f"no free disk space at {resolved}")
    except (OSError, RuntimeError) as exc:
        return _failed("Disk", exc)
    return CheckResult(
        "Disk",
        True,
        f"{usage.free / _GIB:.2f} GiB free of {usage.total / _GIB:.2f} GiB at {resolved}",
    )


def run_doctor(*, gpu_id: int, disk_path: Path) -> DoctorReport:
    """Run all static production-runtime checks in dependency order."""
    gpu, cuda, vram = check_cuda(gpu_id)
    checks = (
        check_docker(),
        check_driver(gpu_id),
        gpu,
        cuda,
        check_tensorrt(),
        check_driver_library("NVDEC", "libnvcuvid.so.1", "cuvidCreateDecoder"),
        check_driver_library("NVENC", "libnvidia-encode.so.1", "NvEncodeAPICreateInstance"),
        check_pynvvideocodec(),
        check_cvcuda(),
        vram,
        check_disk(disk_path),
    )
    return DoctorReport(checks=checks)
