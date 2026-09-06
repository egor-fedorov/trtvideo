from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.scripts.contracts.manifest import ManifestContractError, hardware_environment
from benchmarks.scripts.runtime import environment
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


def test_hardware_collector_matches_run_identity_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(environment, "_package_version", lambda _: "version")
    monkeypatch.setattr(environment, "_command_version", lambda _: "version")
    monkeypatch.setattr(environment, "_cuda_runtime_version", lambda: "13.2")
    gpu = {"name": "Test GPU", "power_limit_w": 350.0, "driver_version": "595.84"}

    hardware = environment.collect_hardware_environment(gpu)
    assert hardware == hardware_environment(environment.collect_environment(gpu))
    assert hardware == {"gpu": gpu, "cpu": {"model": "Test CPU", "logical_cores": 12}}


def test_hardware_probe_uses_exact_image_and_selected_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware = {"gpu": {"index": 1, "name": "GPU"}, "cpu": {"model": "CPU"}}
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(hardware))

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.probe_hardware_environment(image_id="sha256:image", gpu_id=1) == hardware
    assert calls == [
        (
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "none",
                "sha256:image",
                "python3",
                "-m",
                "benchmarks.scripts.runtime.environment",
                "--gpu-id",
                "1",
            ],
            {"check": True, "capture_output": True, "text": True, "timeout": 60},
        )
    ]


@pytest.mark.parametrize("stdout", ["not-json", "[]", "{}", '{"gpu": {}, "cpu": {}}'])
def test_hardware_probe_rejects_missing_or_malformed_contract(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, stdout=stdout),
    )
    with pytest.raises((ValueError, ManifestContractError)):
        environment.probe_hardware_environment(image_id="sha256:image", gpu_id=0)


@pytest.mark.parametrize("fail", [False, True])
def test_hardware_probe_cli_releases_nvml_without_starting_sampling(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fail: bool
) -> None:
    calls = []

    class Sampler:
        def __init__(self, gpu_id):
            calls.append(gpu_id)

        def initialize(self):
            calls.append("initialize")
            if fail:
                raise RuntimeError("driver unavailable")
            return {"index": 1, "name": "GPU"}

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(environment, "NvmlSampler", Sampler)
    monkeypatch.setattr(environment.sys, "argv", ["environment", "--gpu-id", "1"])
    if fail:
        with pytest.raises(SystemExit) as exc:
            environment.main()
        assert exc.value.code == 2
        assert "driver unavailable" in capsys.readouterr().err
    else:
        environment.main()
        assert json.loads(capsys.readouterr().out)["gpu"] == {"index": 1, "name": "GPU"}
    assert calls == [1, "initialize", "shutdown"]
