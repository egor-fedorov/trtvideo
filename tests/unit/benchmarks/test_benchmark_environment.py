from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.scripts.runtime.environment import _cuda_runtime_version, collect_image_identity


def test_image_identity_records_base_image_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "registry.example/runtime:version@sha256:" + "a" * 64
    monkeypatch.setenv("TRTVIDEO_BASE_IMAGE", reference)

    assert collect_image_identity()["base_reference"] == reference


def test_image_identity_reports_unknown_base_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTVIDEO_BASE_IMAGE", raising=False)

    assert collect_image_identity()["base_reference"] == "unknown"


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
