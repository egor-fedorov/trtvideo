#!/usr/bin/env python3
"""Build a pinned VSGAN TensorRT 10 engine and write a parity sidecar."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import sha256_file, write_json
from benchmarks.scripts.runners.common import (
    CompetitorError,
    find_model_variant,
    find_variant,
    load_json,
)

TENSORRT_VERSION_RE = re.compile(r"TensorRT[^\n]*v([0-9]+)", re.IGNORECASE)


def build_command(
    *,
    onnx_path: Path,
    engine_path: Path,
    timing_cache: Path | None,
) -> list[str]:
    """Return the strongly typed pinned-runtime trtexec build command."""
    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--stronglyTyped",
        "--builderOptimizationLevel=5",
        "--memPoolSize=workspace:8192MiB",
    ]
    if timing_cache is not None:
        command.append(f"--timingCacheFile={timing_cache}")
    return command


def _run_and_log(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    lines.append(line)
            returncode = process.wait()
    except OSError as exc:
        raise CompetitorError(f"Cannot execute trtexec: {exc}") from exc
    if returncode != 0:
        raise CompetitorError(f"trtexec exited with code {returncode}; see {log_path}")
    return "".join(lines)


def _version_from_output(output: str) -> str:
    match = TENSORRT_VERSION_RE.search(output)
    return match.group(1) if match else "unknown"


def write_sidecar(
    *,
    manifest: dict[str, Any],
    variant_name: str,
    onnx_path: Path,
    engine_path: Path,
    timing_cache: Path | None,
    build_log: Path,
    command: list[str],
    trtexec_output: str,
) -> Path:
    """Write the engine contract consumed by the VSGAN benchmark runner."""
    model_variant = find_model_variant(manifest, variant_name)
    clip_variant = find_variant(manifest, variant_name)
    output = clip_variant["benchmark_output"]
    sidecar_path = Path(f"{engine_path}.json")
    sidecar = {
        "schema_version": 1,
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "onnx_path": str(onnx_path),
        "model_sha256": sha256_file(onnx_path),
        "onnx_opset": None,
        "tensorrt_version": _version_from_output(trtexec_output),
        "builder_base_image": os.environ.get("AI_MEDIA_BASE_IMAGE", "unknown"),
        "precision": "mixed-fp16",
        "io_precision": "fp32",
        "input": {
            "name": "input",
            "shape": [
                1,
                3,
                model_variant["input_height"],
                model_variant["input_width"],
            ],
            "dtype": "float32",
        },
        "output": {
            "name": "output",
            "shape": [1, 3, output["height"], output["width"]],
            "dtype": "float32",
        },
        "input_profile": None,
        "builder_flags": ["stronglyTyped"],
        "builder_optimization_level": 5,
        "workspace_mib": 8192,
        "timing_cache": str(timing_cache) if timing_cache else None,
        "build_log": str(build_log),
        "build_command": command,
        "preprocess_version": "limited_bt709_yuv420p_to_rgb_float32",
        "postprocess_version": "rgb_float32_to_limited_bt709_yuv420p",
        "batch_size": 1,
        "full_frame": True,
        "tiling": False,
    }
    write_json(sidecar_path, sidecar)
    return sidecar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-cache")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = load_json(Path(args.manifest))
        model_variant = find_model_variant(manifest, args.variant)
        onnx_path = Path(args.onnx)
        expected_onnx = Path(model_variant["fp16_path"])
        if onnx_path.resolve() != expected_onnx.resolve():
            raise CompetitorError(
                f"Expected canonical ONNX {expected_onnx}, got {onnx_path}"
            )
        if not onnx_path.is_file() and not args.dry_run:
            raise CompetitorError(f"ONNX not found: {onnx_path}")
        engine_path = Path(args.output)
        timing_cache = Path(args.timing_cache) if args.timing_cache else None
        command = build_command(
            onnx_path=onnx_path,
            engine_path=engine_path,
            timing_cache=timing_cache,
        )
        if args.dry_run:
            print(shlex.join(command))
            return
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        if timing_cache is not None:
            timing_cache.parent.mkdir(parents=True, exist_ok=True)
        build_log = Path(f"{engine_path}.build.log")
        output = _run_and_log(command, build_log)
        if not engine_path.is_file():
            raise CompetitorError("trtexec completed without creating an engine")
        sidecar = write_sidecar(
            manifest=manifest,
            variant_name=args.variant,
            onnx_path=onnx_path,
            engine_path=engine_path,
            timing_cache=timing_cache,
            build_log=build_log,
            command=command,
            trtexec_output=output,
        )
        print(f"VSGAN engine contract: {sidecar}")
    except (CompetitorError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
