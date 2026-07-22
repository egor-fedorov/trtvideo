from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.build_vsgan_engine import (
    build_command as build_vsgan_engine_command,
)
from benchmarks.scripts.competitor_common import (
    CompetitorError,
    benchmark_parameters,
    validate_static_engine_contract,
)
from benchmarks.scripts.external_video_suite import run_command_spec
from benchmarks.scripts.run_trtexec import (
    build_plan as build_trtexec_plan,
)
from benchmarks.scripts.run_trtexec import (
    build_trtexec_command,
    parse_trtexec_output,
)
from benchmarks.scripts.run_vsgan import (
    build_plan as build_vsgan_plan,
)
from benchmarks.scripts.run_vsgan import (
    build_vsgan_command,
)
from benchmarks.scripts.run_vstrt import build_plan as build_vstrt_plan

MANIFEST_PATH = "benchmarks/workloads/realesrgan_x2plus_sintel.json"
IMPLEMENTATIONS_PATH = "benchmarks/implementations.json"


def manifest() -> dict:
    return json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))


def common_args(**overrides) -> argparse.Namespace:
    values = {
        "manifest": MANIFEST_PATH,
        "implementations": IMPLEMENTATIONS_PATH,
        "variant": "1080p",
        "engine": "/app/models/model.engine",
        "input": "/app/videos/benchmarks/sintel_1080p24_h264.mp4",
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
        "mode": "parity",
        "vs_threads": 8,
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


def test_static_engine_contract_checks_onnx_and_bindings(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"canonical onnx")
    sidecar = {
        "model_sha256": hashlib.sha256(b"canonical onnx").hexdigest(),
        "io_precision": "fp32",
        "input_profile": None,
        "input": {"shape": [1, 3, 1080, 1920]},
        "output": {"shape": [1, 3, 2160, 3840]},
    }

    validate_static_engine_contract(sidecar, manifest(), "1080p", onnx_path)

    sidecar["io_precision"] = "fp16"
    with pytest.raises(CompetitorError, match="FP32"):
        validate_static_engine_contract(sidecar, manifest(), "1080p", onnx_path)


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

    assert plan["comparison_class"] == "diagnostic"
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

    plan, _ = build_vstrt_plan(args)
    spec = plan["commands"]["measured"]

    assert len(spec) == 2
    assert spec[0][0] == "vspipe"
    assert spec[0][spec[0].index("--end") + 1] == "999"
    assert "source=/app/videos/benchmarks/sintel_1080p24_h264.mp4" in spec[0]
    assert spec[1][0] == "ffmpeg"
    assert spec[1][spec[1].index("-b:v") + 1] == "60M"
    assert "-bf" in spec[1]
    assert plan["parameters"]["max_compute_processes"] == 2
    assert plan["parameters"]["max_graphics_processes"] == 0
    assert plan["comparison_class"] == "parity"
    assert plan["parameters"]["num_streams"] == 1
    assert plan["parameters"]["batch_size"] == 1
    assert plan["parameters"]["tiling"] is False


def test_vsgan_command_uses_stock_script_and_explicit_nvenc_contract() -> None:
    args = common_args()

    spec = build_vsgan_command(
        args,
        manifest(),
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )
    vspipe, ffmpeg = spec

    assert vspipe[vspipe.index("--requests") + 1] == "1"
    assert "num_streams=1" in vspipe
    assert "cuda_graph=0" in vspipe
    assert vspipe[-2] == "/app/benchmarks/vsgan/upscale.vpy"
    assert ffmpeg[ffmpeg.index("-rc") + 1] == "cbr"
    assert ffmpeg[ffmpeg.index("-b:v") + 1] == "60M"
    assert ffmpeg[ffmpeg.index("-bf") + 1] == "0"


def test_vsgan_plan_is_pinned_product_parity() -> None:
    plan, _ = build_vsgan_plan(common_args())

    assert plan["comparison_class"] == "product"
    assert plan["implementation"]["exact_model_match"] is True
    assert plan["implementation"]["exact_engine_match"] is False
    assert plan["parameters"]["mode"] == "parity"
    assert plan["parameters"]["num_streams"] == 1
    assert plan["parameters"]["max_compute_processes"] == 2
    assert plan["parameters"]["max_graphics_processes"] == 0


def test_vsgan_engine_build_is_static_strongly_typed() -> None:
    command = build_vsgan_engine_command(
        onnx_path=Path("/app/models/model.onnx"),
        engine_path=Path("/app/models/model.engine"),
        timing_cache=Path("/app/models/cache/trt10.cache"),
    )

    assert "--stronglyTyped" in command
    assert "--builderOptimizationLevel=5" in command
    assert "--memPoolSize=workspace:8192MiB" in command
    assert "--skipInference" in command
    assert "--timingCacheFile=/app/models/cache/trt10.cache" in command


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
