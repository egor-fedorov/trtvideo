"""Model-space capture contracts and tensor comparison."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import environment_errors, sha256_file

CAPTURE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 2
TENSOR_DTYPE = "float32"
TENSOR_LAYOUT = "CHW"
TENSOR_CHANNEL_ORDER = "RGB"
TENSOR_STAGES = ("input", "output")


class ModelSpaceError(RuntimeError):
    """Raised when model-space evidence is missing or incompatible."""


@dataclass(frozen=True)
class TensorThresholds:
    """Fixed acceptance limits for one model-space tensor stage."""

    p99_abs: float
    rmse: float
    min_psnr_db: float

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, stage: str) -> TensorThresholds:
        try:
            thresholds = cls(
                p99_abs=float(value["p99_abs"]),
                rmse=float(value["rmse"]),
                min_psnr_db=float(value["min_psnr_db"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelSpaceError(
                f"Invalid model-space thresholds for {stage}: {exc}"
            ) from exc
        if (
            thresholds.p99_abs <= 0
            or thresholds.rmse <= 0
            or thresholds.min_psnr_db <= 0
        ):
            raise ModelSpaceError(
                f"Model-space thresholds for {stage} must be positive"
            )
        return thresholds

    def as_dict(self) -> dict[str, float]:
        return {
            "p99_abs": self.p99_abs,
            "rmse": self.rmse,
            "min_psnr_db": self.min_psnr_db,
        }


@dataclass(frozen=True)
class TensorArtifact:
    """One raw planar float tensor captured at a selected video frame."""

    stage: str
    frame_index: int
    shape: tuple[int, int, int]
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorArtifact:
        try:
            shape_value = value["shape"]
            if not isinstance(shape_value, list):
                raise TypeError("shape must be an array")
            tensor_fields = {
                "dtype": value["dtype"],
                "layout": value["layout"],
                "channel_order": value["channel_order"],
            }
            expected_fields = {
                "dtype": TENSOR_DTYPE,
                "layout": TENSOR_LAYOUT,
                "channel_order": TENSOR_CHANNEL_ORDER,
            }
            if tensor_fields != expected_fields:
                raise ValueError("tensor contract fields do not match the capture")
            shape_values = tuple(int(item) for item in shape_value)
            if len(shape_values) != 3:
                raise ValueError("shape must contain three dimensions")
            artifact = cls(
                stage=str(value["stage"]),
                frame_index=int(value["frame_index"]),
                shape=(shape_values[0], shape_values[1], shape_values[2]),
                path=str(value["path"]),
                sha256=str(value["sha256"]),
                size_bytes=int(value["size_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelSpaceError(f"Invalid tensor artifact: {exc}") from exc
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.stage not in TENSOR_STAGES:
            raise ModelSpaceError(f"Unknown tensor stage: {self.stage}")
        if self.frame_index < 0:
            raise ModelSpaceError("Tensor frame index must be non-negative")
        if len(self.shape) != 3 or self.shape[0] != 3 or any(
            dimension <= 0 for dimension in self.shape
        ):
            raise ModelSpaceError(f"Tensor must use positive RGB CHW shape: {self.shape}")
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ModelSpaceError("Tensor artifact path must stay inside its capture directory")
        expected_size = math.prod(self.shape) * 4
        if self.size_bytes != expected_size:
            raise ModelSpaceError(
                f"Tensor size does not match float32 shape: {self.size_bytes} != {expected_size}"
            )
        if len(self.sha256) != 64:
            raise ModelSpaceError("Tensor artifact SHA256 is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "frame_index": self.frame_index,
            "shape": list(self.shape),
            "dtype": TENSOR_DTYPE,
            "layout": TENSOR_LAYOUT,
            "channel_order": TENSOR_CHANNEL_ORDER,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class CaptureManifest:
    """Complete model-space capture from one implementation."""

    implementation: str
    comparison_class: str
    workload_id: str
    variant: str
    input_sha256: str
    onnx_sha256: str
    engine_sha256: str
    image: dict[str, str]
    artifacts: tuple[TensorArtifact, ...]

    @classmethod
    def load(cls, path: Path) -> CaptureManifest:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelSpaceError(f"Cannot read capture manifest {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelSpaceError(f"Capture manifest must be a JSON object: {path}")
        if value.get("schema_version") != CAPTURE_SCHEMA_VERSION:
            raise ModelSpaceError(f"Unsupported capture schema in {path}")
        tensor_contract = value.get("tensor_contract")
        expected_contract = {
            "dtype": TENSOR_DTYPE,
            "layout": TENSOR_LAYOUT,
            "channel_order": TENSOR_CHANNEL_ORDER,
        }
        if tensor_contract != expected_contract:
            raise ModelSpaceError(f"Unexpected tensor contract in {path}")
        try:
            artifacts_value = value["artifacts"]
            if not isinstance(artifacts_value, list):
                raise TypeError("artifacts must be an array")
            image_value = value["environment"]["image"]
            if not isinstance(image_value, dict):
                raise TypeError("environment.image must be an object")
            manifest = cls(
                implementation=str(value["implementation"]),
                comparison_class=str(value["comparison_class"]),
                workload_id=str(value["workload_id"]),
                variant=str(value["variant"]),
                input_sha256=str(value["assets"]["input_sha256"]),
                onnx_sha256=str(value["assets"]["onnx_sha256"]),
                engine_sha256=str(value["assets"]["engine_sha256"]),
                image={
                    str(key): str(item)
                    for key, item in image_value.items()
                },
                artifacts=tuple(
                    TensorArtifact.from_dict(artifact) for artifact in artifacts_value
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelSpaceError(f"Invalid capture manifest {path}: {exc}") from exc
        manifest.validate(path.parent)
        return manifest

    def validate(self, root: Path) -> None:
        if not self.implementation or not self.comparison_class:
            raise ModelSpaceError("Capture implementation identity is required")
        if not self.workload_id or not self.variant:
            raise ModelSpaceError("Capture workload identity is required")
        identity_errors = environment_errors({"image": self.image})
        if identity_errors:
            raise ModelSpaceError(
                "Capture image identity is not publishable: "
                + "; ".join(identity_errors)
            )
        for name, checksum in (
            ("input", self.input_sha256),
            ("ONNX", self.onnx_sha256),
            ("engine", self.engine_sha256),
        ):
            if len(checksum) != 64:
                raise ModelSpaceError(f"Capture {name} SHA256 is invalid")
        if not self.artifacts:
            raise ModelSpaceError("Capture manifest has no tensors")
        keys = [(artifact.stage, artifact.frame_index) for artifact in self.artifacts]
        if len(keys) != len(set(keys)):
            raise ModelSpaceError("Capture manifest contains duplicate tensor artifacts")
        frame_indices = sorted({artifact.frame_index for artifact in self.artifacts})
        expected_keys = {
            (stage, frame_index)
            for stage in TENSOR_STAGES
            for frame_index in frame_indices
        }
        if set(keys) != expected_keys:
            raise ModelSpaceError("Capture must contain input and output for every frame")
        for artifact in self.artifacts:
            artifact_path = root / artifact.path
            if not artifact_path.is_file():
                raise ModelSpaceError(f"Captured tensor not found: {artifact_path}")
            if artifact_path.stat().st_size != artifact.size_bytes:
                raise ModelSpaceError(f"Captured tensor size changed: {artifact_path}")
            if sha256_file(artifact_path) != artifact.sha256:
                raise ModelSpaceError(f"Captured tensor SHA256 changed: {artifact_path}")

    def artifact_map(self) -> dict[tuple[str, int], TensorArtifact]:
        return {
            (artifact.stage, artifact.frame_index): artifact
            for artifact in self.artifacts
        }


def parse_frame_indices(value: str, *, frame_count: int) -> tuple[int, ...]:
    """Parse sorted unique zero-based frame indexes."""
    try:
        indices = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise ModelSpaceError("Frame indices must be comma-separated integers") from exc
    if not indices:
        raise ModelSpaceError("At least one frame index is required")
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise ModelSpaceError(
            f"Frame indices must stay in [0, {frame_count - 1}], got {indices}"
        )
    return indices


def create_tensor_artifact(
    *,
    stage: str,
    frame_index: int,
    shape: tuple[int, int, int],
    path: Path,
    root: Path,
) -> TensorArtifact:
    """Create a verified artifact record for an existing raw float32 file."""
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ModelSpaceError(f"Tensor artifact is outside capture directory: {path}") from exc
    artifact = TensorArtifact(
        stage=stage,
        frame_index=frame_index,
        shape=shape,
        path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )
    artifact.validate()
    return artifact


def write_capture_manifest(
    path: Path,
    *,
    implementation: str,
    comparison_class: str,
    workload_id: str,
    variant: str,
    input_sha256: str,
    onnx_sha256: str,
    engine_sha256: str,
    image: dict[str, str],
    artifacts: list[TensorArtifact],
) -> None:
    """Write one deterministic model-space capture manifest."""
    value = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "document_type": "model-space-capture",
        "implementation": implementation,
        "comparison_class": comparison_class,
        "workload_id": workload_id,
        "variant": variant,
        "tensor_contract": {
            "dtype": TENSOR_DTYPE,
            "layout": TENSOR_LAYOUT,
            "channel_order": TENSOR_CHANNEL_ORDER,
        },
        "assets": {
            "input_sha256": input_sha256,
            "onnx_sha256": onnx_sha256,
            "engine_sha256": engine_sha256,
        },
        "environment": {"image": image},
        "artifacts": [artifact.as_dict() for artifact in artifacts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_float32_tensors(
    reference_path: Path,
    candidate_path: Path,
    *,
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Compare two raw CHW float32 tensors."""
    import numpy as np

    count = math.prod(shape)
    reference = np.memmap(reference_path, dtype="<f4", mode="r", shape=(count,))
    candidate = np.memmap(candidate_path, dtype="<f4", mode="r", shape=(count,))
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return {
            "elements": count,
            "finite": False,
            "exact": False,
            "mae": None,
            "rmse": None,
            "p99_abs": None,
            "max_abs": None,
            "psnr_db": None,
        }
    absolute_error = np.abs(reference - candidate)
    mse = float(np.mean(np.square(absolute_error, dtype=np.float64)))
    rmse = math.sqrt(mse)
    return {
        "elements": count,
        "finite": finite,
        "exact": mse == 0,
        "mae": float(np.mean(absolute_error, dtype=np.float64)),
        "rmse": rmse,
        "p99_abs": float(np.percentile(absolute_error, 99)),
        "max_abs": float(np.max(absolute_error)),
        "psnr_db": None if mse == 0 else 10 * math.log10(1.0 / mse),
    }


def evaluate_metrics(
    metrics: dict[str, Any],
    thresholds: TensorThresholds,
) -> list[str]:
    """Return deterministic acceptance errors for one tensor comparison."""
    errors = []
    if metrics["finite"] is not True:
        errors.append("tensor contains NaN or infinity")
        return errors
    psnr_db = (
        math.inf if metrics.get("exact") is True else float(metrics["psnr_db"])
    )
    checks = {
        "p99_abs": (float(metrics["p99_abs"]), thresholds.p99_abs, "<="),
        "rmse": (float(metrics["rmse"]), thresholds.rmse, "<="),
        "psnr_db": (psnr_db, thresholds.min_psnr_db, ">="),
    }
    for name, (actual, limit, operator) in checks.items():
        failed = actual > limit if operator == "<=" else actual < limit
        if failed:
            errors.append(f"{name} must be {operator} {limit:g}, got {actual:g}")
    return errors


def compare_captures(
    reference_path: Path,
    candidate_paths: list[Path],
    *,
    thresholds: dict[str, TensorThresholds],
) -> dict[str, Any]:
    """Compare complete captures and return a publication-gate report."""
    reference = CaptureManifest.load(reference_path)
    reference_artifacts = reference.artifact_map()
    frame_indices = sorted(
        {artifact.frame_index for artifact in reference.artifacts}
    )
    comparisons = []
    report_errors: list[str] = []
    for candidate_path in candidate_paths:
        candidate = CaptureManifest.load(candidate_path)
        identity_checks = {
            "workload": (candidate.workload_id, reference.workload_id),
            "variant": (candidate.variant, reference.variant),
            "input SHA256": (candidate.input_sha256, reference.input_sha256),
            "ONNX SHA256": (candidate.onnx_sha256, reference.onnx_sha256),
            "repository revision": (
                candidate.image["repository_revision"],
                reference.image["repository_revision"],
            ),
            "tensor set": (
                set(candidate.artifact_map()),
                set(reference_artifacts),
            ),
        }
        candidate_errors = [
            f"{label} differs"
            for label, (actual, expected) in identity_checks.items()
            if actual != expected
        ]
        if candidate.comparison_class == "parity" and (
            candidate.engine_sha256 != reference.engine_sha256
        ):
            candidate_errors.append("parity engine SHA256 differs")

        tensors = []
        candidate_artifacts = candidate.artifact_map()
        if not candidate_errors:
            for key in sorted(reference_artifacts, key=lambda item: (item[1], item[0])):
                reference_artifact = reference_artifacts[key]
                candidate_artifact = candidate_artifacts[key]
                if candidate_artifact.shape != reference_artifact.shape:
                    candidate_errors.append(
                        f"{key[0]} frame {key[1]} tensor shape differs"
                    )
                    continue
                metrics = compare_float32_tensors(
                    reference_path.parent / reference_artifact.path,
                    candidate_path.parent / candidate_artifact.path,
                    shape=reference_artifact.shape,
                )
                errors = evaluate_metrics(metrics, thresholds[key[0]])
                tensors.append(
                    {
                        "stage": key[0],
                        "frame_index": key[1],
                        "shape": list(reference_artifact.shape),
                        "metrics": metrics,
                        "status": "valid" if not errors else "invalid",
                        "errors": errors,
                    }
                )
                candidate_errors.extend(
                    f"{key[0]} frame {key[1]}: {error}" for error in errors
                )

        comparisons.append(
            {
                "implementation": candidate.implementation,
                "comparison_class": candidate.comparison_class,
                "engine_sha256": candidate.engine_sha256,
                "image": candidate.image,
                "status": "valid" if not candidate_errors else "invalid",
                "errors": candidate_errors,
                "tensors": tensors,
            }
        )
        report_errors.extend(
            f"{candidate.implementation}: {error}" for error in candidate_errors
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "document_type": "model-space-parity",
        "status": "valid" if not report_errors else "invalid",
        "publishable": not report_errors,
        "workload_id": reference.workload_id,
        "variant": reference.variant,
        "frame_indices": frame_indices,
        "reference": {
            "implementation": reference.implementation,
            "engine_sha256": reference.engine_sha256,
            "image": reference.image,
        },
        "assets": {
            "input_sha256": reference.input_sha256,
            "onnx_sha256": reference.onnx_sha256,
        },
        "thresholds": {
            stage: stage_thresholds.as_dict()
            for stage, stage_thresholds in thresholds.items()
        },
        "comparisons": comparisons,
        "errors": report_errors,
    }
