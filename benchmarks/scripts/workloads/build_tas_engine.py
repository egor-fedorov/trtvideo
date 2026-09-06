#!/usr/bin/env python3
"""Build a canonical workload engine with the pinned native TAS builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.scripts.contracts.benchmark import (
    CompetitorError,
    implementation_config,
    load_json,
)
from benchmarks.scripts.contracts.engine import (
    EngineContractError,
    validate_tas_engine_contract,
)
from benchmarks.scripts.runtime.environment import sha256_file, write_json
from benchmarks.scripts.workloads.manifest import WorkloadError, find_model_variant

IMPLEMENTATIONS_PATH = Path(__file__).resolve().parents[2] / "implementations.json"


def build_tas_engine(
    *,
    manifest_path: Path,
    variant_name: str,
    onnx_path: Path,
    engine_path: Path,
    gpu_id: int,
) -> Path:
    """Build outside measurement, then persist validated native build evidence."""
    manifest = load_json(manifest_path)
    model_variant = find_model_variant(manifest, variant_name)
    expected_onnx = Path(model_variant["fp16_path"])
    if onnx_path.resolve() != expected_onnx.resolve():
        raise CompetitorError(f"Expected canonical ONNX {expected_onnx}, got {onnx_path}")
    if not onnx_path.is_file():
        raise CompetitorError(f"ONNX not found: {onnx_path}")
    if gpu_id < 0:
        raise CompetitorError("GPU id cannot be negative")
    metadata = implementation_config(load_json(IMPLEMENTATIONS_PATH), "tas")

    # Heavy upstream imports belong to the TAS image, never the host orchestrator.
    from benchmarks.scripts.runners.tas_runtime import build_engine

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = build_engine(
        onnx=onnx_path,
        engine=engine_path,
        width=model_variant["input_width"],
        height=model_variant["input_height"],
        gpu_id=gpu_id,
    )
    if not engine_path.is_file():
        raise EngineContractError("TAS native builder completed without creating an engine")
    if sidecar.get("schema_version") != 1:
        raise EngineContractError("TAS native builder returned an unsupported sidecar schema")
    if sidecar.get("engine_sha256") != sha256_file(engine_path):
        raise EngineContractError("TAS native builder returned a mismatched engine SHA256")
    validate_tas_engine_contract(sidecar, manifest, variant_name, onnx_path, metadata)
    sidecar.update({"engine_path": str(engine_path), "onnx_path": str(onnx_path)})
    sidecar_path = Path(f"{engine_path}.json")
    write_json(sidecar_path, sidecar)
    return sidecar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        sidecar_path = build_tas_engine(
            manifest_path=Path(args.manifest),
            variant_name=args.variant,
            onnx_path=Path(args.onnx),
            engine_path=Path(args.output),
            gpu_id=args.gpu_id,
        )
        print(f"TheAnimeScripter engine contract: {sidecar_path}")
    except (
        CompetitorError,
        EngineContractError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        WorkloadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
