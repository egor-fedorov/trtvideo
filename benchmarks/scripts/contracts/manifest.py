"""Canonical loading and identity checks for benchmark run manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_PARAMETER_KEYS = (
    "mode",
    "vspipe_requests",
    "num_streams",
    "vapoursynth_threads",
    "cuda_graph",
)


class ManifestContractError(RuntimeError):
    """Raised when benchmark evidence violates the shared manifest contract."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from benchmark evidence."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestContractError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestContractError(f"Expected a JSON object in {path}")
    return value


def artifact_path(root: Path, value: Any, *, label: str) -> Path:
    """Resolve a repository-relative artifact without allowing path escape."""
    if not isinstance(value, str) or not value:
        raise ManifestContractError(f"{label} has no artifact path")
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ManifestContractError(f"{label} escapes the repository root")
    return path


def asset_sha(
    manifest: dict[str, Any],
    name: str,
    *,
    exact_length: int | None = None,
) -> str:
    """Extract one required asset checksum."""
    value = manifest.get("assets", {}).get(name, {}).get("sha256")
    if not isinstance(value, str) or not value:
        raise ManifestContractError(f"Manifest has no SHA256 for {name}")
    if exact_length is not None and len(value) != exact_length:
        raise ManifestContractError(
            f"Manifest {name} SHA256 must contain {exact_length} characters"
        )
    return value


def benchmark_contract_version(manifest: dict[str, Any]) -> int:
    """Extract a positive workload benchmark contract version."""
    value = manifest.get("benchmark_contract_version")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestContractError(
            "Manifest has no valid benchmark contract version"
        )
    return value


def execution_profile(parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract all scheduling fields from runner parameters."""
    missing = [key for key in PROFILE_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise ManifestContractError(
            "Manifest has no execution profile fields: " + ", ".join(missing)
        )
    return {key: parameters[key] for key in PROFILE_PARAMETER_KEYS}


def expected_comparison_class(implementation: str, mode: str) -> str:
    """Return the comparison class fixed by one execution profile."""
    if mode != "parity":
        return mode
    return "parity" if implementation == "vstrt" else "single-stream-parity"


def validate_execution_profile(
    manifest: dict[str, Any],
    *,
    implementation: str,
    expected_mode: str,
    expected_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate scheduling fields and comparison class together."""
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ManifestContractError("Manifest has no parameters")
    profile = execution_profile(parameters)
    if profile["mode"] != expected_mode:
        raise ManifestContractError(
            f"{implementation} execution profile is {profile['mode']!r}, "
            f"expected {expected_mode!r}"
        )
    comparison_class = expected_comparison_class(implementation, expected_mode)
    if manifest.get("comparison_class") != comparison_class:
        raise ManifestContractError(
            f"{implementation} comparison class does not match "
            f"{expected_mode} execution profile"
        )
    if expected_values is not None and profile != expected_values:
        raise ManifestContractError(
            f"{implementation} changed execution profile"
        )
    return profile


@dataclass(frozen=True)
class RunIdentity:
    """Immutable identity shared by campaign and tuning validation."""

    workload_id: str
    variant: str
    benchmark_contract_version: int
    input_sha256: str
    onnx_sha256: str
    engine_sha256: str
    workload_sha256: str | None
    image_id: str
    repository_revision: str
    frames: int
    warmup_frames: int | None
    encoder: dict[str, Any]

    def shared_model_key(self) -> tuple[Any, ...]:
        """Return fields that every implementation must share."""
        return (
            self.workload_id,
            self.variant,
            self.benchmark_contract_version,
            self.input_sha256,
            self.onnx_sha256,
            self.workload_sha256,
            self.repository_revision,
            self.frames,
            self.warmup_frames,
            json.dumps(self.encoder, sort_keys=True),
        )

    def implementation_key(self) -> tuple[Any, ...]:
        """Return shared fields plus implementation-specific engine/image."""
        return (*self.shared_model_key(), self.engine_sha256, self.image_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "variant": self.variant,
            "benchmark_contract_version": self.benchmark_contract_version,
            "input_sha256": self.input_sha256,
            "onnx_sha256": self.onnx_sha256,
            "engine_sha256": self.engine_sha256,
            "workload_sha256": self.workload_sha256,
            "image_id": self.image_id,
            "repository_revision": self.repository_revision,
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "encoder": self.encoder,
        }


@dataclass(frozen=True)
class RunExpectation:
    """Optional values required from a run in a specific validation context."""

    product: str | None = None
    workload_id: str | None = None
    variant: str | None = None
    benchmark_contract_version: int | None = None
    run_index: int | None = None
    implementation: str | None = None
    execution_profile: dict[str, Any] | None = None
    require_reproducible: bool = True
    require_media_validation: bool = False
    require_workload_identity: bool = True
    require_warmup_frames: bool = True


def extract_run_identity(
    manifest: dict[str, Any],
    *,
    checksum_length: int | None = None,
    require_workload_identity: bool = True,
    require_warmup_frames: bool = True,
) -> RunIdentity:
    """Extract and structurally validate immutable run fields."""
    parameters = manifest.get("parameters")
    image = manifest.get("environment", {}).get("image")
    if not isinstance(parameters, dict):
        raise ManifestContractError("Run manifest has no parameters")
    if not isinstance(image, dict):
        raise ManifestContractError("Run manifest has no image identity")
    encoder = parameters.get("encoder")
    if not isinstance(encoder, dict):
        raise ManifestContractError("Run manifest has no encoder contract")
    workload_id = manifest.get("workload_id")
    variant = manifest.get("variant")
    image_id = image.get("id")
    revision = image.get("repository_revision")
    frames = parameters.get("frames")
    warmup_frames = parameters.get("warmup_frames")
    if not isinstance(workload_id, str) or not workload_id:
        raise ManifestContractError("Run manifest has no workload id")
    if not isinstance(variant, str) or not variant:
        raise ManifestContractError("Run manifest has no variant")
    if not isinstance(image_id, str) or not image_id:
        raise ManifestContractError("Run manifest has no image id")
    if not isinstance(revision, str) or not revision:
        raise ManifestContractError("Run manifest has no repository revision")
    if str(image.get("source_dirty")) != "0":
        raise ManifestContractError("Run manifest was built from dirty source")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise ManifestContractError(
            "Run manifest has invalid measured frame count"
        )
    if require_warmup_frames:
        if (
            not isinstance(warmup_frames, int)
            or isinstance(warmup_frames, bool)
            or warmup_frames <= 0
        ):
            raise ManifestContractError(
                "Run manifest has invalid warmup frame count"
            )
    elif warmup_frames is not None and (
        not isinstance(warmup_frames, int)
        or isinstance(warmup_frames, bool)
        or warmup_frames < 0
    ):
        raise ManifestContractError("Run manifest has invalid warmup frame count")
    workload_sha256 = (
        asset_sha(
            manifest,
            "workload_manifest",
            exact_length=checksum_length,
        )
        if require_workload_identity
        else None
    )
    return RunIdentity(
        workload_id=workload_id,
        variant=variant,
        benchmark_contract_version=benchmark_contract_version(manifest),
        input_sha256=asset_sha(
            manifest,
            "input",
            exact_length=checksum_length,
        ),
        onnx_sha256=asset_sha(
            manifest,
            "onnx",
            exact_length=checksum_length,
        ),
        engine_sha256=asset_sha(
            manifest,
            "engine",
            exact_length=checksum_length,
        ),
        workload_sha256=workload_sha256,
        image_id=image_id,
        repository_revision=revision,
        frames=frames,
        warmup_frames=warmup_frames,
        encoder=encoder,
    )


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    expectation: RunExpectation,
    checksum_length: int | None = None,
) -> RunIdentity:
    """Validate one run and return its immutable identity."""
    if manifest.get("status") != "valid":
        raise ManifestContractError("Run manifest status is not valid")
    checks = {
        "product": (manifest.get("product"), expectation.product),
        "workload": (manifest.get("workload_id"), expectation.workload_id),
        "variant": (manifest.get("variant"), expectation.variant),
        "benchmark contract version": (
            manifest.get("benchmark_contract_version"),
            expectation.benchmark_contract_version,
        ),
        "run index": (manifest.get("run_index"), expectation.run_index),
    }
    for label, (actual, expected) in checks.items():
        if expected is not None and actual != expected:
            raise ManifestContractError(f"Run manifest changed {label}")
    if (
        expectation.require_reproducible
        and manifest.get("reproducibility", {}).get("publishable") is not True
    ):
        raise ManifestContractError("Run manifest is not reproducible")
    if (
        expectation.require_media_validation
        and manifest.get("measured", {}).get("validation", {}).get("valid")
        is not True
    ):
        raise ManifestContractError(
            "Run manifest failed complete media validation"
        )
    if expectation.execution_profile is not None:
        if expectation.implementation is None:
            raise ManifestContractError(
                "Execution profile validation requires an implementation"
            )
        validate_execution_profile(
            manifest,
            implementation=expectation.implementation,
            expected_mode=str(expectation.execution_profile["mode"]),
            expected_values=expectation.execution_profile,
        )
    return extract_run_identity(
        manifest,
        checksum_length=checksum_length,
        require_workload_identity=expectation.require_workload_identity,
        require_warmup_frames=expectation.require_warmup_frames,
    )
