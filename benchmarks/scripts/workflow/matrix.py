"""Load and validate the canonical benchmark workload matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class WorkflowMatrixError(RuntimeError):
    """Raised when a workflow matrix is malformed or ambiguous."""


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowMatrixError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise WorkflowMatrixError(f"{label} must stay inside the repository")
    return value


@dataclass(frozen=True)
class Variant:
    """Model artifacts required for one input resolution."""

    name: str
    onnx: str
    engine: str
    vsgan_engine: str


@dataclass(frozen=True)
class Workload:
    """One model workload and all declared resolution variants."""

    key: str
    manifest: str
    tuning_contract: str
    variants: tuple[Variant, ...]

    def variant(self, name: str) -> Variant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise WorkflowMatrixError(
            f"Workload {self.key!r} has no {name!r} variant"
        )


@dataclass(frozen=True)
class Selection:
    """One workload-resolution pair selected for execution."""

    workload: Workload
    variant: Variant

    @property
    def key(self) -> str:
        return f"{self.workload.key}-{self.variant.name}"


@dataclass(frozen=True)
class WorkflowMatrix:
    """Validated canonical matrix and diagnostic selection."""

    workloads: tuple[Workload, ...]
    nsight_workload: str
    nsight_variant: str

    @property
    def workload_keys(self) -> tuple[str, ...]:
        return tuple(workload.key for workload in self.workloads)

    def select(
        self,
        *,
        workload_key: str | None,
        variant_name: str | None,
    ) -> tuple[Selection, ...]:
        selections = []
        for workload in self.workloads:
            if workload_key is not None and workload.key != workload_key:
                continue
            for variant in workload.variants:
                if variant_name is not None and variant.name != variant_name:
                    continue
                selections.append(Selection(workload=workload, variant=variant))
        if not selections:
            raise WorkflowMatrixError("Workflow selection is empty")
        return tuple(selections)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowMatrixError(f"Cannot read workflow matrix {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowMatrixError("Workflow matrix must be a JSON object")
    return value


def load_workflow_matrix(path: Path) -> WorkflowMatrix:
    """Load a strict matrix without accepting implicit artifact paths."""
    document = _load_json(path)
    if document.get("schema_version") != 1:
        raise WorkflowMatrixError("Unsupported workflow matrix schema version")
    workload_values = document.get("workloads")
    if not isinstance(workload_values, list) or not workload_values:
        raise WorkflowMatrixError("Workflow matrix has no workloads")

    workloads = []
    seen_workloads: set[str] = set()
    for workload_value in workload_values:
        if not isinstance(workload_value, dict):
            raise WorkflowMatrixError("Workflow entry must be an object")
        key = workload_value.get("key")
        if not isinstance(key, str) or not key:
            raise WorkflowMatrixError("Workflow entry has no key")
        if key in seen_workloads:
            raise WorkflowMatrixError(f"Duplicate workflow key: {key}")
        seen_workloads.add(key)
        variants_value = workload_value.get("variants")
        if not isinstance(variants_value, dict) or not variants_value:
            raise WorkflowMatrixError(f"Workload {key!r} has no variants")
        variants = []
        for name, variant_value in variants_value.items():
            if name not in {"720p", "1080p"} or not isinstance(variant_value, dict):
                raise WorkflowMatrixError(
                    f"Workload {key!r} has invalid variant {name!r}"
                )
            variants.append(
                Variant(
                    name=name,
                    onnx=_relative_path(
                        variant_value.get("onnx"),
                        label=f"{key}.{name}.onnx",
                    ),
                    engine=_relative_path(
                        variant_value.get("engine"),
                        label=f"{key}.{name}.engine",
                    ),
                    vsgan_engine=_relative_path(
                        variant_value.get("vsgan_engine"),
                        label=f"{key}.{name}.vsgan_engine",
                    ),
                )
            )
        workloads.append(
            Workload(
                key=key,
                manifest=_relative_path(
                    workload_value.get("manifest"),
                    label=f"{key}.manifest",
                ),
                tuning_contract=_relative_path(
                    workload_value.get("tuning_contract"),
                    label=f"{key}.tuning_contract",
                ),
                variants=tuple(variants),
            )
        )

    diagnostics = document.get("diagnostics", {}).get("nsight")
    if not isinstance(diagnostics, dict):
        raise WorkflowMatrixError("Workflow matrix has no Nsight selection")
    nsight_workload = diagnostics.get("workload")
    nsight_variant = diagnostics.get("variant")
    if not isinstance(nsight_workload, str) or not isinstance(nsight_variant, str):
        raise WorkflowMatrixError("Workflow matrix has an invalid Nsight selection")
    matrix = WorkflowMatrix(
        workloads=tuple(workloads),
        nsight_workload=nsight_workload,
        nsight_variant=nsight_variant,
    )
    matrix.select(
        workload_key=nsight_workload,
        variant_name=nsight_variant,
    )
    return matrix
