from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path

import pytest

from benchmarks.scripts.runners.trtvideo import build_command
from benchmarks.scripts.runners.trtvideo_suite import (
    PRODUCT_NAME,
    BenchmarkConfig,
    BenchmarkError,
    build_trtvideo_command,
    validate_config,
)
from benchmarks.scripts.runtime.environment import relative_artifact_path, sanitize_command
from benchmarks.scripts.runtime.suite import (
    SuitePolicy,
    SuiteRunner,
    canonical_suite_errors,
    compute_suite_statistics,
    report_invalid_run,
    report_publishability_errors,
    should_extend_suite,
    suite_publishability_errors,
)


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


def test_build_trtvideo_command_has_no_profiling_flags(tmp_path: Path) -> None:
    benchmark = config(tmp_path)

    command = build_trtvideo_command(
        benchmark,
        output_path=tmp_path / "output.mp4",
        frame_count=1000,
        lifecycle_path=tmp_path / "lifecycle.json",
    )

    assert command[:2] == ["trtvideo", "--engine"]
    assert "--profile" not in command
    assert "--profile-json" not in command
    assert command[command.index("--max-frames") + 1] == "1000"
    assert command[command.index("--bitrate-mbps") + 1] == "35.0"
    assert command[command.index("--benchmark-lifecycle-json") + 1].endswith("lifecycle.json")


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


def test_suite_runner_applies_extension_idle_and_power_limit_invariant() -> None:
    values = [10.0, 12.0, 8.0, 10.0, 10.0]
    sleeps: list[float] = []

    def run(index: int) -> dict:
        return {
            "status": "valid",
            "run_index": index,
            "metric": values[index - 1],
            "power_limit": 250.0,
        }

    runner = SuiteRunner(
        SuitePolicy(3, 2, 0.05, 10),
        label="test",
        frames=1000,
        metric_reader=lambda manifest: manifest["metric"],
        power_limit_reader=lambda manifest: manifest["power_limit"],
        sleep=sleeps.append,
    )

    result = runner.execute(run)

    assert result.status == "unstable"
    assert result.target_runs == 5
    assert len(result.runs) == 5
    assert sleeps == [10, 10, 10, 10]


def test_suite_runner_rejects_power_limit_drift() -> None:
    def run(index: int) -> dict:
        return {
            "status": "valid",
            "run_index": index,
            "metric": 10.0,
            "power_limit": 250.0 if index < 3 else 300.0,
        }

    result = SuiteRunner(
        SuitePolicy(3, 0, 0.05, 0),
        label="test",
        frames=1000,
        metric_reader=lambda manifest: manifest["metric"],
        power_limit_reader=lambda manifest: manifest["power_limit"],
    ).execute(run)

    assert result.status == "invalid"
    assert result.errors == ("GPU power limit changed between measured runs",)


def test_suite_runner_can_extend_at_one_percent_and_accept_below_five() -> None:
    values = [100.0, 102.0, 101.0, 101.5, 100.5]
    runner = SuiteRunner(
        SuitePolicy(3, 2, 0.01, 0, max_relative_spread=0.05),
        label="test",
        frames=1000,
        metric_reader=lambda manifest: manifest["metric"],
        power_limit_reader=lambda manifest: manifest["power_limit"],
    )

    result = runner.execute(
        lambda index: {
            "status": "valid",
            "run_index": index,
            "metric": values[index - 1],
            "power_limit": 250.0,
        }
    )

    assert result.target_runs == 5
    assert result.status == "valid"


def test_single_run_suite_uses_product_name_without_progress_fraction() -> None:
    output = StringIO()
    runner = SuiteRunner(
        SuitePolicy(1, 0, 0.05, 0),
        label=PRODUCT_NAME,
        frames=1000,
        metric_reader=lambda manifest: manifest["metric"],
        power_limit_reader=lambda manifest: manifest["power_limit"],
        stream=output,
    )

    runner.execute(
        lambda index: {
            "status": "valid",
            "run_index": index,
            "metric": 40.0,
            "power_limit": 250.0,
        }
    )

    assert output.getvalue() == "Benchmark: trtvideo, 1000 frames\n"


def test_invalid_run_reports_manifest_errors(capsys: pytest.CaptureFixture[str]) -> None:
    report_invalid_run(
        {
            "run_index": 2,
            "errors": ["Warmup process exited with code 1", "Output was not created"],
        }
    )

    assert capsys.readouterr().err == (
        "Benchmark run 2 invalid:\n"
        "  - Warmup process exited with code 1\n"
        "  - Output was not created\n"
    )


def test_acceptance_suite_suppresses_expected_publishability_errors() -> None:
    output = StringIO()

    report_publishability_errors(
        ["Individual suites are acceptance-only"],
        acceptance_only=True,
        stream=output,
    )

    assert output.getvalue() == ""


def test_standalone_suite_reports_publishability_errors() -> None:
    output = StringIO()

    report_publishability_errors(
        ["Run contract changed"],
        acceptance_only=False,
        stream=output,
    )

    assert output.getvalue() == ("Benchmark suite is not publishable:\n  - Run contract changed\n")


def test_smoke_parameters_are_valid_but_not_canonical() -> None:
    benchmark = {
        "warmup_frames": 100,
        "measured_frames": 1000,
        "initial_runs": 3,
        "extra_runs_on_spread": 2,
        "spread_threshold": 0.05,
        "max_relative_spread": 0.05,
        "idle_seconds": 10,
        "nvml_sample_interval_ms": 100,
    }
    parameters = {
        "warmup_frames": 24,
        "frames": 120,
        "initial_runs": 1,
        "extra_runs_on_spread": 0,
        "spread_threshold": 0.05,
        "max_relative_spread": 0.05,
        "idle_seconds": 0,
        "nvml_sample_interval_ms": 100,
    }

    errors = canonical_suite_errors(
        parameters,
        benchmark,
        include_warmup_frames=True,
    )

    assert errors == [
        "frames must match canonical measured_frames (120 != 1000)",
        "initial_runs must match canonical initial_runs (1 != 3)",
        "extra_runs_on_spread must match canonical extra_runs_on_spread (0 != 2)",
        "idle_seconds must match canonical idle_seconds (0 != 10)",
        "warmup_frames must match canonical warmup_frames (24 != 100)",
    ]


def test_canonical_parameters_are_publishable() -> None:
    benchmark = {
        "warmup_frames": 100,
        "measured_frames": 1000,
        "initial_runs": 3,
        "extra_runs_on_spread": 2,
        "spread_threshold": 0.05,
        "max_relative_spread": 0.05,
        "idle_seconds": 10,
        "nvml_sample_interval_ms": 100,
    }
    parameters = {
        "warmup_frames": 100,
        "frames": 1000,
        "initial_runs": 3,
        "extra_runs_on_spread": 2,
        "spread_threshold": 0.05,
        "max_relative_spread": 0.05,
        "idle_seconds": 10,
        "nvml_sample_interval_ms": 100,
    }

    assert (
        canonical_suite_errors(
            parameters,
            benchmark,
            include_warmup_frames=True,
        )
        == []
    )


def test_individual_suite_is_acceptance_only() -> None:
    errors = suite_publishability_errors(
        status="valid",
        canonical_errors=[],
        runs=[],
        acceptance_only=True,
    )

    assert errors == ["Individual suites are acceptance-only; use a rotated campaign result"]


def test_sanitize_command_does_not_leak_external_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    internal = root / "models" / "model.engine"
    external = tmp_path / "private" / "input.mp4"

    sanitized = sanitize_command(["trtvideo", str(internal), str(external)], root)

    assert sanitized == ["trtvideo", "models/model.engine", "external/input.mp4"]
    assert relative_artifact_path(external, root) == "external/input.mp4"


def test_canonical_runner_consumes_manifest_contract() -> None:
    manifest = json.loads(
        Path("benchmarks/workloads/realesrgan_x2plus_sintel.json").read_text(encoding="utf-8")
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
        keep_outputs=False,
        skip_bitrate_validation=False,
    )

    command = build_command(args, manifest)

    assert command[0] == "benchmark-trtvideo"
    assert command[command.index("--input") + 1] == ("videos/benchmarks/sintel_1080p24_h264.mp4")
    assert command[command.index("--bitrate-mbps") + 1] == "60"
    assert command[command.index("--warmup-frames") + 1] == "30"
    assert command[command.index("--frames") + 1] == "1000"
    assert command[command.index("--runs") + 1] == "3"


def test_canonical_runner_preserves_explicit_zero_for_downstream_validation() -> None:
    manifest = json.loads(
        Path("benchmarks/workloads/realesrgan_x2plus_sintel.json").read_text(encoding="utf-8")
    )
    args = argparse.Namespace(
        manifest="benchmarks/workloads/realesrgan_x2plus_sintel.json",
        variant="1080p",
        engine="models/model.engine",
        output_dir="artefacts/results",
        json=None,
        gpu_id=0,
        frames=0,
        warmup_frames=0,
        runs=0,
        extra_runs=None,
        idle_seconds=None,
        keep_outputs=False,
        skip_bitrate_validation=False,
    )

    command = build_command(args, manifest)

    assert command[command.index("--warmup-frames") + 1] == "0"
    assert command[command.index("--frames") + 1] == "0"
    assert command[command.index("--runs") + 1] == "0"


def test_canonical_runner_can_skip_bitrate_validation_for_smoke() -> None:
    manifest = json.loads(
        Path("benchmarks/workloads/realesrgan_x2plus_sintel.json").read_text(encoding="utf-8")
    )
    args = argparse.Namespace(
        manifest="benchmarks/workloads/realesrgan_x2plus_sintel.json",
        variant="720p",
        engine="models/model.engine",
        output_dir="artefacts/results",
        json=None,
        gpu_id=0,
        frames=120,
        warmup_frames=24,
        runs=1,
        extra_runs=0,
        idle_seconds=0,
        keep_outputs=False,
        skip_bitrate_validation=True,
    )

    command = build_command(args, manifest)

    assert "--skip-bitrate-validation" in command
