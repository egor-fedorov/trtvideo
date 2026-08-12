from __future__ import annotations

from typing import Any

import pytest

from benchmarks.scripts.contracts.manifest import (
    ManifestContractError,
    RunExpectation,
    validate_run_manifest,
)


def _manifest() -> dict[str, Any]:
    return {
        "status": "valid",
        "product": "vs-mlrt",
        "workload_id": "workload-v1",
        "variant": "1080p",
        "benchmark_contract_version": 2,
        "run_index": 1,
        "parameters": {
            "frames": 400,
            "warmup_frames": 30,
            "encoder": {"codec": "h264"},
            "execution_profile": "tuned",
            "vspipe_requests": 4,
            "num_streams": 2,
            "vapoursynth_threads": 8,
            "cuda_graph": False,
        },
        "assets": {
            "input": {"sha256": "input"},
            "onnx": {"sha256": "onnx"},
            "engine": {"sha256": "engine"},
            "workload_manifest": {"sha256": "workload"},
        },
        "environment": {
            "gpu": {
                "name": "NVIDIA GeForce RTX 3090",
                "driver_version": "595.84",
                "power_limit_w": 350.0,
            },
            "cpu": {"model": "Test CPU", "logical_cores": 12},
            "image": {
                "id": "image",
                "repository_revision": "revision",
                "source_dirty": "0",
            },
        },
        "reproducibility": {"publishable": True},
        "measured": {"validation": {"valid": True}},
    }


def test_validate_run_manifest_returns_complete_performance_identity() -> None:
    manifest = _manifest()

    identity = validate_run_manifest(
        manifest,
        expectation=RunExpectation(
            product="vs-mlrt",
            workload_id="workload-v1",
            variant="1080p",
            benchmark_contract_version=2,
            run_index=1,
            implementation="vstrt",
            execution_profile={
                "execution_profile": "tuned",
                "vspipe_requests": 4,
                "num_streams": 2,
                "vapoursynth_threads": 8,
                "cuda_graph": False,
            },
            require_media_validation=True,
        ),
    )

    assert identity.workload_sha256 == "workload"
    assert identity.warmup_frames == 30
    assert identity.environment["gpu"]["power_limit_w"] == 350.0


@pytest.mark.parametrize("key", ["gpu", "cpu"])
def test_validate_run_manifest_requires_hardware_contract(key: str) -> None:
    manifest = _manifest()
    manifest["environment"].pop(key)

    with pytest.raises(ManifestContractError, match=key.upper()):
        validate_run_manifest(
            manifest,
            expectation=RunExpectation(require_hardware_environment=True),
        )


def test_quality_run_can_omit_performance_only_identity_fields() -> None:
    manifest = _manifest()
    manifest["parameters"].pop("warmup_frames")
    manifest["assets"].pop("workload_manifest")

    identity = validate_run_manifest(
        manifest,
        expectation=RunExpectation(
            product="vs-mlrt",
            require_media_validation=True,
            require_workload_identity=False,
            require_warmup_frames=False,
        ),
    )

    assert identity.workload_sha256 is None
    assert identity.warmup_frames is None


def test_validate_run_manifest_rejects_execution_profile_drift() -> None:
    manifest = _manifest()
    manifest["parameters"]["num_streams"] = 3

    with pytest.raises(ManifestContractError, match="changed execution profile"):
        validate_run_manifest(
            manifest,
            expectation=RunExpectation(
                implementation="vstrt",
                execution_profile={
                    "execution_profile": "tuned",
                    "vspipe_requests": 4,
                    "num_streams": 2,
                    "vapoursynth_threads": 8,
                    "cuda_graph": False,
                },
            ),
        )
