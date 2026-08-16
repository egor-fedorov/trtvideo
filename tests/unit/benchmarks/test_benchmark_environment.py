from __future__ import annotations

from types import SimpleNamespace

from benchmarks.scripts.runtime.environment import _cuda_runtime_version


def _runtime(*, status: int = 0, version: int = 13020):
    return SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaRuntimeGetVersion=lambda: (status, version),
    )


def test_cuda_runtime_version_uses_cuda_bindings_without_torch() -> None:
    runtime = _runtime()

    assert _cuda_runtime_version(importer=lambda _name: runtime) == "13.2"


def test_cuda_runtime_version_preserves_patch_component() -> None:
    runtime = _runtime(version=12031)

    assert _cuda_runtime_version(importer=lambda _name: runtime) == "12.3.1"


def test_cuda_runtime_version_is_optional_when_query_fails() -> None:
    runtime = _runtime(status=1)

    assert _cuda_runtime_version(importer=lambda _name: runtime) is None
