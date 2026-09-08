from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.contracts.benchmark import (
    CompetitorError,
    benchmark_parameters,
)
from benchmarks.scripts.runners.trtexec import (
    build_plan as build_trtexec_plan,
)
from benchmarks.scripts.runners.trtexec import (
    build_trtexec_command,
    parse_trtexec_output,
)
from benchmarks.scripts.runners.vapoursynth_profile import (
    add_execution_profile_arguments,
)
from benchmarks.scripts.runners.vapoursynth_suite import (
    run_command_spec,
    suite_implementation_parameters,
)
from benchmarks.scripts.runners.vstrt import (
    build_plan as build_vstrt_plan,
)
from benchmarks.scripts.runners.vstrt import (
    build_vstrt_command,
)

MANIFEST_PATH = "benchmarks/workloads/realesrgan_x2plus_madrid.json"
IMPLEMENTATIONS_PATH = "benchmarks/implementations.json"
BENCHMARK_PLAN_KEYS = {
    "schema_version",
    "document_type",
    "product",
    "backend",
    "workload_id",
    "benchmark_contract_version",
    "variant",
    "implementation",
    "parameters",
    "commands",
    "assets",
    "limitations",
}


def manifest() -> dict:
    return json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))


def common_args(**overrides) -> argparse.Namespace:
    values = {
        "manifest": MANIFEST_PATH,
        "implementations": IMPLEMENTATIONS_PATH,
        "variant": "1080p",
        "engine": "/app/models/model.engine",
        "input": "/app/videos/benchmarks/madrid_2021_05_06_1080p24_h264.mp4",
        "output_dir": "/app/artefacts/results",
        "json": None,
        "gpu_id": 0,
        "frames": None,
        "warmup_frames": None,
        "runs": None,
        "extra_runs": None,
        "spread_threshold": None,
        "max_relative_spread": None,
        "idle_seconds": None,
        "dry_run": True,
        "cuda_graph": None,
        "warmup_ms": 1000,
        "requests": None,
        "num_streams": None,
        "execution_profile": "upstream-default",
        "vs_threads": None,
        "skip_bitrate_validation": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_shared_parameters_accept_smoke_overrides() -> None:
    args = common_args(frames=120, warmup_frames=24, runs=1, extra_runs=0)

    parameters = benchmark_parameters(args, manifest())

    assert parameters["frames"] == 120
    assert parameters["warmup_frames"] == 24
    assert parameters["initial_runs"] == 1
    assert parameters["extra_runs_on_spread"] == 0


def test_shared_parameters_separate_extension_and_acceptance_spread() -> None:
    args = common_args(spread_threshold=0.01, max_relative_spread=0.05)

    parameters = benchmark_parameters(args, manifest())

    assert parameters["spread_threshold"] == 0.01
    assert parameters["max_relative_spread"] == 0.05


def test_external_suite_records_disabled_bitrate_acceptance() -> None:
    plan, _ = build_vstrt_plan(common_args(skip_bitrate_validation=True))

    parameters = suite_implementation_parameters(plan["parameters"])

    assert parameters["bitrate_validation"] is False
    assert parameters["execution_profile"] == "upstream-default"
    assert parameters["encoder"]["rate_control"] == "cbr"


def test_removed_single_request_profile_is_not_accepted() -> None:
    parser = argparse.ArgumentParser()
    add_execution_profile_arguments(parser)

    assert parser.parse_args([]).execution_profile == "upstream-default"
    with pytest.raises(SystemExit):
        parser.parse_args(["--execution-profile", "parity"])


def test_trtexec_command_explicitly_matches_cuda_graph_mode() -> None:
    args = common_args(cuda_graph=False)

    command = build_trtexec_command(
        args,
        export_times=Path("/app/artefacts/times.json"),
        iterations=1000,
    )

    assert "--iterations=1000" in command
    assert "--duration=0" in command
    assert "--noCudaGraph" in command
    assert not any(value == "--noDataTransfers" for value in command)


def test_trtexec_plan_is_diagnostic() -> None:
    args = common_args()

    plan, _ = build_trtexec_plan(args)

    assert plan["implementation"]["role"] == "diagnostic"
    assert plan["benchmark_contract_version"] == 2
    assert "--iterations=1000" in plan["commands"]["measured"][0]
    assert plan["parameters"]["data_transfers"] is False
    assert plan["assets"][0]["present"] is False


def test_parse_trtexec_output() -> None:
    output = """
    Throughput: 41.25 qps
    Latency: min = 22.0 ms, max = 30.0 ms, mean = 24.5 ms,
      median = 24.1 ms, percentile(50%) = 24.1 ms,
      percentile(95%) = 27.3 ms, percentile(99%) = 29.4 ms
    GPU Compute Time: min = 20.0 ms, max = 28.0 ms, mean = 23.4 ms,
      median = 23.1 ms, percentile(50%) = 23.1 ms,
      percentile(95%) = 26.2 ms, percentile(99%) = 27.8 ms
    """

    metrics = parse_trtexec_output(output)

    assert metrics == {
        "throughput_qps": 41.25,
        "latency_median_ms": 24.1,
        "latency_p95_ms": 27.3,
        "gpu_compute_median_ms": 23.1,
        "gpu_compute_p95_ms": 26.2,
    }


def test_vstrt_plan_uses_absolute_container_input() -> None:
    args = common_args()
    original_input = args.input

    plan, _ = build_vstrt_plan(args)
    spec = plan["commands"]["measured"]

    assert len(spec) == 2
    assert spec[0][0] == "vspipe"
    assert spec[0][spec[0].index("--end") + 1] == "999"
    assert "source=/app/videos/benchmarks/madrid_2021_05_06_1080p24_h264.mp4" in spec[0]
    assert spec[1][0] == "ffmpeg"
    assert spec[1][spec[1].index("-b:v") + 1] == "60000000"
    assert spec[1][spec[1].index("-rc_init_occupancy") + 1] == "60000000"
    assert spec[1][spec[1].index("-multipass") + 1] == "disabled"
    assert "-bf" in spec[1]
    assert plan["parameters"]["max_compute_processes"] == 2
    assert plan["parameters"]["max_graphics_processes"] == 0
    assert plan["parameters"]["execution_profile"] == "upstream-default"
    assert set(plan) == BENCHMARK_PLAN_KEYS
    assert plan["parameters"]["vspipe_requests"] == "auto"
    assert plan["parameters"]["num_streams"] == 1
    assert plan["parameters"]["batch_size"] == 1
    assert plan["parameters"]["tiling"] is False
    assert args.input == original_input


def test_vapoursynth_script_accepts_runtime_default_threads() -> None:
    source = Path("benchmarks/vstrt/upscale.vpy").read_text(encoding="utf-8")

    assert 'configured_threads = globals().get("vs_threads")' in source
    assert "if configured_threads is not None:" in source


def test_external_smoke_plan_can_skip_bitrate_validation() -> None:
    plan, _ = build_vstrt_plan(common_args(skip_bitrate_validation=True))

    assert plan["parameters"]["bitrate_validation"] is False


def test_vstrt_upstream_default_uses_automatic_vspipe_requests() -> None:
    args = common_args(
        execution_profile="upstream-default",
        requests=None,
        num_streams=None,
        cuda_graph=None,
    )

    plan, benchmark_manifest = build_vstrt_plan(args)
    vspipe, _ = build_vstrt_command(
        args,
        benchmark_manifest,
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )

    assert "--requests" not in vspipe
    assert "num_streams=1" in vspipe
    assert not any(value.startswith("vs_threads=") for value in vspipe)
    assert plan["parameters"]["execution_profile"] == "upstream-default"
    assert (
        plan["commands"]["measured"][0][plan["commands"]["measured"][0].index("--end") + 1] == "999"
    )
    assert plan["parameters"]["vspipe_requests"] == "auto"
    assert plan["parameters"]["vapoursynth_threads"] == "auto"


def test_tuned_profile_requires_explicit_scheduling_contract() -> None:
    with pytest.raises(CompetitorError, match="tuned requires explicit"):
        build_vstrt_plan(
            common_args(
                execution_profile="tuned",
                requests=None,
                num_streams=None,
                vs_threads=None,
                cuda_graph=None,
            )
        )


def test_tuned_profile_records_explicit_scheduling_contract() -> None:
    args = common_args(
        execution_profile="tuned",
        requests="auto",
        num_streams=3,
        vs_threads=6,
        cuda_graph=True,
    )

    plan, benchmark_manifest = build_vstrt_plan(args)
    vspipe, _ = build_vstrt_command(
        args,
        benchmark_manifest,
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )

    assert "--requests" not in vspipe
    assert "num_streams=3" in vspipe
    assert "vs_threads=6" in vspipe
    assert "cuda_graph=1" in vspipe
    assert plan["parameters"]["execution_profile"] == "tuned"
    assert plan["parameters"]["cuda_graph"] is True


def test_command_pipeline_executes_without_shell(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    spec = [
        [sys.executable, "-c", "import sys; sys.stdout.write('abc')"],
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
    ]

    result = run_command_spec(spec, stdout, stderr)

    assert result.returncode == 0
    assert stdout.read_text(encoding="utf-8") == "ABC"
    assert stderr.read_text(encoding="utf-8") == ""


def test_command_pipeline_observes_vspipe_progress(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    spec = [
        [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "sys.stderr.write('Frame: 1/2\\r'); sys.stderr.flush(); "
                "time.sleep(0.05); "
                "sys.stdout.write('abc')"
            ),
        ],
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
    ]

    result = run_command_spec(
        spec,
        stdout,
        stderr,
        observe_vspipe_progress=True,
    )

    assert result.returncode == 0
    assert result.first_frame_completed_ns is not None
    assert result.producer_finished_ns is not None
    assert result.process_started_ns <= result.first_frame_completed_ns
    assert result.first_frame_completed_ns <= result.producer_finished_ns
    assert result.producer_finished_ns <= result.process_finished_ns
