from __future__ import annotations

from typing import Any

import pytest

from benchmarks.scripts.contracts.manifest import (
    ManifestContractError,
    RunExpectation,
    execution_profile,
    hardware_environment,
    validate_execution_profile,
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
    assert identity.environment == hardware_environment(manifest["environment"])


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


def _tas_profile() -> dict[str, Any]:
    return {
        "execution_profile": "tuned",
        "decode_method": "nvdec",
        "writer": "nelux",
        "cuda_graph": True,
    }


@pytest.mark.parametrize("decode_method", ["cpu", "nvdec"])
@pytest.mark.parametrize("writer", ["ffmpeg", "nelux"])
def test_tas_profile_extracts_only_io_fields(decode_method: str, writer: str) -> None:
    profile = {**_tas_profile(), "decode_method": decode_method, "writer": writer}
    parameters = {**profile, "frames": 1000, "encoder": {"codec": "h264"}}

    assert execution_profile(parameters) == profile


@pytest.mark.parametrize("graph", [False, None, 0, 1, "true"])
def test_tas_profile_requires_enabled_cuda_graph(graph: Any) -> None:
    with pytest.raises(ManifestContractError, match="requires cuda_graph=true"):
        execution_profile({**_tas_profile(), "cuda_graph": graph})


@pytest.mark.parametrize("missing", ["execution_profile", "decode_method", "writer", "cuda_graph"])
def test_tas_profile_requires_all_fields(missing: str) -> None:
    profile = _tas_profile()
    profile.pop(missing)

    with pytest.raises(ManifestContractError, match="no execution profile fields"):
        execution_profile(profile)


@pytest.mark.parametrize("key", ["vspipe_requests", "num_streams", "vapoursynth_threads"])
def test_tas_profile_rejects_vapoursynth_fields(key: str) -> None:
    with pytest.raises(ManifestContractError, match="mixes TAS and VapourSynth"):
        execution_profile({**_tas_profile(), key: 1})


@pytest.mark.parametrize("key,value", [("decode_method", "auto"), ("writer", "pipe")])
def test_tas_profile_rejects_unknown_io_method(key: str, value: str) -> None:
    with pytest.raises(ManifestContractError, match=key):
        execution_profile({**_tas_profile(), key: value})


@pytest.mark.parametrize("implementation", ["tas", "TheAnimeScripter"])
def test_tas_identity_rejects_vapoursynth_profile(implementation: str) -> None:
    with pytest.raises(ManifestContractError, match="requires a TAS execution profile"):
        validate_execution_profile(
            _manifest(), implementation=implementation, expected_profile="tuned"
        )


def test_vstrt_identity_rejects_tas_profile() -> None:
    with pytest.raises(ManifestContractError, match="requires a VapourSynth execution profile"):
        validate_execution_profile(
            {"parameters": _tas_profile()}, implementation="vstrt", expected_profile="tuned"
        )


def test_tas_run_identity_keeps_common_model_and_separate_engine() -> None:
    manifest = _manifest()
    manifest["product"] = "TheAnimeScripter"
    manifest["parameters"] = {
        "frames": 400,
        "warmup_frames": 30,
        "encoder": {"codec": "h264"},
        **_tas_profile(),
    }
    manifest["assets"]["engine"]["sha256"] = "tas-engine"
    identity = validate_run_manifest(
        manifest,
        expectation=RunExpectation(
            product="TheAnimeScripter",
            implementation="tas",
            execution_profile=_tas_profile(),
            require_media_validation=True,
        ),
    )

    assert identity.onnx_sha256 == "onnx"
    assert identity.engine_sha256 == "tas-engine"


def test_tas_run_profile_rejects_writer_fallback() -> None:
    with pytest.raises(ManifestContractError, match="changed execution profile"):
        validate_execution_profile(
            {"parameters": {**_tas_profile(), "writer": "ffmpeg"}},
            implementation="tas",
            expected_profile="tuned",
            expected_values=_tas_profile(),
        )
