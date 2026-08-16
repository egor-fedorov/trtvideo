from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trtvideo.cli.doctor import render_report
from trtvideo.diagnostics.doctor import (
    CheckResult,
    DoctorReport,
    check_cuda,
    check_docker,
    check_driver,
    check_driver_library,
    check_pynvvideocodec,
)


class _FakeCudaRuntime:
    class cudaError_t:
        cudaSuccess = 0

    @staticmethod
    def cudaGetDeviceCount() -> tuple[int, int]:
        return 0, 1

    @staticmethod
    def cudaSetDevice(device_id: int) -> tuple[int]:
        assert device_id == 0
        return (0,)

    @staticmethod
    def cudaGetDeviceProperties(device_id: int) -> tuple[int, SimpleNamespace]:
        assert device_id == 0
        return 0, SimpleNamespace(name=b"Test GPU\0ignored")

    @staticmethod
    def cudaRuntimeGetVersion() -> tuple[int, int]:
        return 0, 13020

    @staticmethod
    def cudaDriverGetVersion() -> tuple[int, int]:
        return 0, 13010

    @staticmethod
    def cudaMemGetInfo() -> tuple[int, int, int]:
        return 0, 8 * 1024**3, 24 * 1024**3


def test_cuda_check_reports_gpu_versions_and_vram() -> None:
    def importer(name: str) -> object:
        assert name == "cuda.bindings.runtime"
        return _FakeCudaRuntime

    gpu, cuda, vram = check_cuda(0, importer=importer)

    assert gpu == CheckResult("GPU", True, "device 0: Test GPU")
    assert cuda == CheckResult("CUDA", True, "runtime 13.2, driver API 13.1")
    assert vram == CheckResult("VRAM", True, "8.00 GiB free of 24.00 GiB")


def test_cuda_check_rejects_unavailable_gpu() -> None:
    results = check_cuda(1, importer=lambda _name: _FakeCudaRuntime)

    assert all(not result.passed for result in results)
    assert all("GPU 1 is unavailable" in result.detail for result in results)


def test_docker_marker_avoids_daemon_probe(tmp_path: Path) -> None:
    marker = tmp_path / ".dockerenv"
    marker.touch()

    result = check_docker(
        marker=marker,
        command_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )

    assert result == CheckResult("Docker", True, "container detected")


def test_driver_version_comes_from_nvidia_smi() -> None:
    completed = SimpleNamespace(returncode=0, stdout="595.84\n", stderr="")

    result = check_driver(0, command_runner=lambda *_args, **_kwargs: completed)

    assert result == CheckResult("Driver", True, "595.84")


def test_video_driver_library_requires_public_symbol() -> None:
    result = check_driver_library(
        "NVENC",
        "libnvidia-encode.so.1",
        "NvEncodeAPICreateInstance",
        loader=lambda _name: SimpleNamespace(NvEncodeAPICreateInstance=object()),
    )

    assert result.passed


def test_pynvvideocodec_requires_production_api() -> None:
    module = SimpleNamespace(ThreadedDecoder=object(), CreateEncoder=object(), __version__="2.1")

    result = check_pynvvideocodec(importer=lambda _name: module)

    assert result == CheckResult("PyNvVideoCodec", True, "2.1")


def test_report_is_nonzero_ready_only_when_all_checks_pass() -> None:
    report = DoctorReport(
        checks=(
            CheckResult("CUDA", True, "13.2"),
            CheckResult("NVENC", False, "library missing"),
        )
    )

    output = render_report(report)

    assert not report.ready
    assert "[PASS] CUDA" in output
    assert "[FAIL] NVENC" in output
    assert output.endswith("Not ready: 1 required check failed.")
