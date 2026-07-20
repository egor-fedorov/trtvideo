from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.scripts.competitor_common import benchmark_parameters
from benchmarks.scripts.external_video_suite import run_command_spec
from benchmarks.scripts.run_trtexec import (
    build_plan as build_trtexec_plan,
)
from benchmarks.scripts.run_trtexec import (
    build_trtexec_command,
    parse_trtexec_output,
)
from benchmarks.scripts.run_video2x import build_video2x_command
from benchmarks.scripts.run_vstrt import build_vstrt_command

MANIFEST_PATH = "benchmarks/workloads/realesrgan_x2plus_sintel.json"
COMPETITORS_PATH = "benchmarks/competitors.json"


def manifest() -> dict:
    return json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))


def common_args(**overrides) -> argparse.Namespace:
    values = {
        "manifest": MANIFEST_PATH,
        "competitors": COMPETITORS_PATH,
        "variant": "1080p",
        "engine": "/app/models/model.engine",
        "output_dir": "/app/artefacts/results",
        "json": None,
        "gpu_id": 0,
        "frames": None,
        "warmup_frames": None,
        "runs": None,
        "extra_runs": None,
        "idle_seconds": None,
        "dry_run": True,
        "cuda_graph": False,
        "warmup_ms": 1000,
        "requests": 1,
        "num_streams": 1,
        "video2x_model": "realesr-animevideov3",
        "hwaccel": "cuda",
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


def test_trtexec_plan_is_inference_ceiling() -> None:
    args = common_args()

    plan, _ = build_trtexec_plan(args)

    assert plan["comparison_class"] == "inference-ceiling"
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


def test_vstrt_command_is_an_argv_pipeline() -> None:
    args = common_args()
    args.input = "videos/benchmarks/sintel_1080p24_h264.mp4"

    spec = build_vstrt_command(
        args,
        manifest(),
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )

    assert len(spec) == 2
    assert spec[0][0] == "vspipe"
    assert spec[0][spec[0].index("--end") + 1] == "999"
    assert spec[1][0] == "ffmpeg"
    assert spec[1][spec[1].index("-b:v") + 1] == "60M"
    assert "-bf" in spec[1]


def test_video2x_command_is_explicitly_product_level() -> None:
    args = common_args()

    spec = build_video2x_command(
        args,
        manifest(),
        input_path=Path("/app/videos/input.mp4"),
        output_path=Path("/app/artefacts/output.mp4"),
    )
    command = spec[0]

    assert command[command.index("--realesrgan-model") + 1] == "realesr-animevideov3"
    assert command[command.index("--scaling-factor") + 1] == "2"
    assert command[command.index("--bit-rate") + 1] == "60000000"
    assert command[command.index("--max-b-frames") + 1] == "0"


def test_command_pipeline_executes_without_shell(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    spec = [
        [sys.executable, "-c", "import sys; sys.stdout.write('abc')"],
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
    ]

    returncode = run_command_spec(spec, stdout, stderr)

    assert returncode == 0
    assert stdout.read_text(encoding="utf-8") == "ABC"
    assert stderr.read_text(encoding="utf-8") == ""
