from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.runners.common import (
    CompetitorError,
    benchmark_parameters,
    validate_static_engine_contract,
)
from benchmarks.scripts.runners.external_video_suite import run_command_spec
from benchmarks.scripts.runners.trtexec import (
    build_plan as build_trtexec_plan,
)
from benchmarks.scripts.runners.trtexec import (
    build_trtexec_command,
    parse_trtexec_output,
)
from benchmarks.scripts.runners.vsgan import (
    _validate_parity_engine,
    build_vsgan_command,
)
from benchmarks.scripts.runners.vsgan import (
    build_plan as build_vsgan_plan,
)
from benchmarks.scripts.runners.vstrt import (
    build_plan as build_vstrt_plan,
)
from benchmarks.scripts.runners.vstrt import (
    build_vstrt_command,
)
from benchmarks.scripts.workloads.build_vsgan_engine import (
    build_command as build_vsgan_engine_command,
)

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
    assert plan["benchmark_contract_version"] == 2
    assert "--iterations=400" in plan["commands"]["measured"][0]
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
    assert spec[0][spec[0].index("--end") + 1] == "399"
    assert "source=/app/videos/benchmarks/sintel_1080p24_h264.mp4" in spec[0]
    assert spec[1][0] == "ffmpeg"
    assert spec[1][spec[1].index("-b:v") + 1] == "60000000"
    assert spec[1][spec[1].index("-rc_init_occupancy") + 1] == "60000000"
    assert spec[1][spec[1].index("-multipass") + 1] == "disabled"
    assert "-bf" in spec[1]
    assert plan["parameters"]["max_compute_processes"] == 2
    assert plan["parameters"]["max_graphics_processes"] == 0
    assert plan["comparison_class"] == "parity"
    assert plan["parameters"]["num_streams"] == 1
    assert plan["parameters"]["batch_size"] == 1
    assert plan["parameters"]["tiling"] is False
    assert args.input == original_input


def test_vsgan_command_uses_pinned_script_and_explicit_nvenc_contract() -> None:
    args = common_args()

    spec = build_vsgan_command(
        args,
        manifest(),
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )
    vspipe, ffmpeg = spec

    assert vspipe[vspipe.index("--requests") + 1] == "1"
    assert "--progress" in vspipe
    assert "num_streams=1" in vspipe
    assert "cuda_graph=0" in vspipe
    assert not any(value.startswith("vs_threads=") for value in vspipe)
    assert vspipe[-2] == "/app/benchmarks/vsgan/upscale.vpy"
    assert ffmpeg[ffmpeg.index("-rc") + 1] == "cbr"
    assert ffmpeg[ffmpeg.index("-b:v") + 1] == "60000000"
    assert ffmpeg[ffmpeg.index("-bufsize") + 1] == "120000000"
    assert ffmpeg[ffmpeg.index("-rc_init_occupancy") + 1] == "60000000"
    assert ffmpeg[ffmpeg.index("-bf") + 1] == "0"


@pytest.mark.parametrize(
    "script",
    (
        "benchmarks/vstrt/upscale.vpy",
        "benchmarks/vsgan/upscale.vpy",
    ),
)
def test_vapoursynth_scripts_accept_runtime_default_threads(script: str) -> None:
    source = Path(script).read_text(encoding="utf-8")

    assert 'configured_threads = globals().get("vs_threads")' in source
    assert "if configured_threads is not None:" in source


def test_vsgan_plan_is_single_stream_parity() -> None:
    plan, _ = build_vsgan_plan(common_args())
    vspipe, _ = plan["commands"]["measured"]

    assert plan["benchmark_contract_version"] == 2
    assert vspipe[vspipe.index("--end") + 1] == "399"
    assert plan["comparison_class"] == "single-stream-parity"
    assert plan["implementation"]["exact_model_match"] is True
    assert plan["implementation"]["exact_engine_match"] is False
    assert plan["implementation"]["upstream_tag"] == "latest_no_avx512"
    assert plan["implementation"]["encoder_ffmpeg_package"] == "7:6.1.1-3ubuntu5"
    assert plan["parameters"]["mode"] == "parity"
    assert plan["parameters"]["num_streams"] == 1
    assert plan["parameters"]["max_compute_processes"] == 2
    assert plan["parameters"]["max_graphics_processes"] == 0


def test_external_smoke_plan_can_skip_bitrate_validation() -> None:
    plan, _ = build_vstrt_plan(common_args(skip_bitrate_validation=True))

    assert plan["parameters"]["bitrate_validation"] is False


def test_vstrt_upstream_default_uses_automatic_vspipe_requests() -> None:
    args = common_args(
        mode="upstream-default",
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
    assert plan["comparison_class"] == "upstream-default"
    assert plan["commands"]["measured"][0][
        plan["commands"]["measured"][0].index("--end") + 1
    ] == "399"
    assert plan["parameters"]["vspipe_requests"] == "auto"
    assert plan["parameters"]["vapoursynth_threads"] == "auto"


def test_vsgan_upstream_default_matches_pinned_configuration() -> None:
    args = common_args(
        mode="upstream-default",
        requests=None,
        num_streams=None,
        cuda_graph=None,
    )

    plan, benchmark_manifest = build_vsgan_plan(args)
    vspipe, _ = build_vsgan_command(
        args,
        benchmark_manifest,
        output_path=Path("/app/artefacts/output.mp4"),
        frames=1000,
    )

    assert "--requests" not in vspipe
    assert "num_streams=4" in vspipe
    assert "vs_threads=4" in vspipe
    assert plan["comparison_class"] == "upstream-default"
    assert plan["commands"]["measured"][0][
        plan["commands"]["measured"][0].index("--end") + 1
    ] == "399"
    assert plan["implementation"]["role"] == "product"
    assert plan["parameters"]["vspipe_requests"] == "auto"


def test_tuned_profile_requires_explicit_scheduling_contract() -> None:
    with pytest.raises(CompetitorError, match="tuned requires explicit"):
        build_vsgan_plan(
            common_args(
                mode="tuned",
                requests=None,
                num_streams=None,
                vs_threads=None,
                cuda_graph=None,
            )
        )


def test_tuned_profile_records_explicit_scheduling_contract() -> None:
    args = common_args(
        mode="tuned",
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
    assert plan["comparison_class"] == "tuned"
    assert plan["parameters"]["cuda_graph"] is True


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


def test_vsgan_engine_rejects_different_base_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"canonical onnx")
    sidecar = {
        "model_sha256": hashlib.sha256(b"canonical onnx").hexdigest(),
        "io_precision": "fp32",
        "input_profile": None,
        "input": {"shape": [1, 3, 1080, 1920]},
        "output": {"shape": [1, 3, 2160, 3840]},
        "builder_flags": ["stronglyTyped"],
        "tensorrt_version": "101600",
        "builder_base_image": "old-image@sha256:old",
    }
    monkeypatch.setenv("AI_MEDIA_BASE_IMAGE", "new-image@sha256:new")
    monkeypatch.setenv("AI_MEDIA_VSGAN_FFMPEG_PACKAGE", "ffmpeg-version")

    with pytest.raises(CompetitorError, match="different base image"):
        _validate_parity_engine(
            sidecar,
            manifest(),
            "1080p",
            onnx_path,
            "new-image@sha256:new",
            "ffmpeg-version",
        )


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
