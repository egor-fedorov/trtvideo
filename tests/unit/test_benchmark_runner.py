from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ai_media.benchmarking.environment import relative_artifact_path, sanitize_command
from ai_media.benchmarking.runner import (
    BenchmarkConfig,
    BenchmarkError,
    build_upscale_command,
    compute_suite_statistics,
    should_extend_suite,
    validate_config,
)
from benchmarks.scripts.run_ai_media import build_command


def config(tmp_path: Path, bitrate_mbps: float | None = 35.0) -> BenchmarkConfig:
    engine = tmp_path / "model.engine"
    input_path = tmp_path / "input.mp4"
    engine.write_bytes(b"engine")
    input_path.write_bytes(b"video")
    return BenchmarkConfig(
        engine=engine,
        input_path=input_path,
        output_dir=tmp_path / "results",
        bitrate_mbps=bitrate_mbps,
    )


def test_build_upscale_command_has_no_profiling_flags(tmp_path: Path) -> None:
    benchmark = config(tmp_path)

    command = build_upscale_command(
        benchmark,
        output_path=tmp_path / "output.mp4",
        frame_count=1000,
    )

    assert "--profile" not in command
    assert "--profile-json" not in command
    assert command[command.index("--max-frames") + 1] == "1000"
    assert command[command.index("--bitrate-mbps") + 1] == "35.0"


def test_validate_config_requires_explicit_nvenc_bitrate(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="explicit positive"):
        validate_config(config(tmp_path, bitrate_mbps=None))


def test_suite_statistics_and_automatic_extension() -> None:
    stable = [40.0, 40.5, 39.8]
    unstable = [40.0, 44.0, 38.0]

    report = compute_suite_statistics(stable)

    assert report["median_fps"] == 40.0
    assert should_extend_suite(stable, 0.05) is False
    assert should_extend_suite(unstable, 0.05) is True


def test_sanitize_command_does_not_leak_external_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    internal = root / "models" / "model.engine"
    external = tmp_path / "private" / "input.mp4"

    sanitized = sanitize_command(["upscale", str(internal), str(external)], root)

    assert sanitized == ["upscale", "models/model.engine", "external/input.mp4"]
    assert relative_artifact_path(external, root) == "external/input.mp4"


def test_canonical_runner_consumes_manifest_contract() -> None:
    manifest = json.loads(
        Path("benchmarks/workloads/realesrgan_x2plus_sintel.json").read_text(
            encoding="utf-8"
        )
    )
    args = argparse.Namespace(
        manifest="benchmarks/workloads/realesrgan_x2plus_sintel.json",
        variant="1080p",
        engine="models/model.engine",
        output_dir="artefacts/results",
        json=None,
        gpu_id=0,
        frames=None,
        warmup_frames=None,
        runs=None,
        extra_runs=None,
        idle_seconds=None,
        cuda_graph=False,
        keep_outputs=False,
    )

    command = build_command(args, manifest)

    assert command[command.index("--input") + 1] == (
        "videos/benchmarks/sintel_1080p24_h264.mp4"
    )
    assert command[command.index("--bitrate-mbps") + 1] == "60"
    assert command[command.index("--warmup-frames") + 1] == "100"
    assert command[command.index("--frames") + 1] == "1000"
    assert command[command.index("--runs") + 1] == "3"
