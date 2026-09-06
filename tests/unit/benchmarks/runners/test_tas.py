from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.contracts.benchmark import CompetitorError
from benchmarks.scripts.runners import tas, tas_suite
from benchmarks.scripts.runners.tas_process import build_parser as adapter_parser
from benchmarks.scripts.runners.tas_profile import (
    TasExecutionProfile,
    resolve_execution_profile,
    validate_declared_profile,
)
from benchmarks.scripts.runtime import video_suite
from benchmarks.scripts.runtime.command import command_spec
from benchmarks.scripts.runtime.environment import sha256_file, write_json
from benchmarks.scripts.runtime.suite import SuitePolicy
from benchmarks.scripts.runtime.video_suite import ProcessResult
from trtvideo.benchmarking.lifecycle import FrameLifecycleMarkers, write_frame_markers
from trtvideo.video.nvcodec.encoder import NvencCbrContract

MANIFEST = "benchmarks/workloads/liveaction_span_madrid.json"
REVISION = "a" * 40
RUNTIME = {
    "python": "3.14.0",
    "torch": "2.10.0",
    "tensorrt": "11.0.0",
    "nelux": "0.9.0",
    "ffmpeg": "ffmpeg version test",
}


@pytest.fixture
def implementation() -> dict[str, Any]:
    return {
        "role": "product",
        "image": "trtvideo:benchmark-tas",
        "source_revision": REVISION,
        "execution_profiles": {
            "upstream-default": {
                "decode_method": "cpu",
                "writer": "ffmpeg",
                "cuda_graph": True,
            },
        },
    }


@pytest.fixture
def args(tmp_path: Path, implementation: dict[str, Any]) -> argparse.Namespace:
    metadata = tmp_path / "implementations.json"
    write_json(metadata, {"schema_version": 1, "implementations": {"tas": implementation}})
    return tas.build_parser().parse_args(
        [
            "--manifest",
            MANIFEST,
            "--implementations",
            str(metadata),
            "--variant",
            "720p",
            "--engine",
            "/app/models/tas/span.engine",
            "--output-dir",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )


def test_tas_plan_declares_real_media_process_and_normalized_encoder(args) -> None:
    plan, manifest = tas.build_plan(args)
    profile = resolve_execution_profile(args)
    parameters = plan["parameters"]
    command = plan["commands"]["measured"][0]

    assert profile.as_parameters() == {
        "execution_profile": "upstream-default",
        "decode_method": "cpu",
        "writer": "ffmpeg",
        "cuda_graph": True,
    }
    assert plan["product"] == "TheAnimeScripter"
    assert plan["backend"] == "tas"
    assert len(plan["commands"]["measured"]) == 1
    assert command[:3] == [sys.executable, "-m", "benchmarks.scripts.runners.tas_process"]
    assert command[command.index("--input") + 1].startswith("/app/videos/")
    assert command[command.index("--onnx") + 1].startswith("/app/models/")
    assert command[command.index("--engine") + 1] == args.engine
    assert command[command.index("--frames") + 1] == "1000"
    assert command[command.index("--fps") + 1] == manifest["clip"]["fps"]
    assert command[command.index("--encoder-contract") + 1].endswith("encoder.contract.json")
    assert command[command.index("--lifecycle-output") + 1].endswith(".lifecycle.json")
    assert command[command.index("--runtime-evidence") + 1].endswith(".runtime-evidence.json")
    assert (
        parameters["encoder"] == NvencCbrContract(bitrate_bps=35_000_000, gop_frames=24).as_dict()
    )
    assert parameters["max_compute_processes"] == 2
    assert parameters["max_graphics_processes"] == 0
    assert set(tas_suite.suite_implementation_parameters(parameters)) == {
        "execution_profile",
        "decode_method",
        "writer",
        "cuda_graph",
        "batch_size",
        "full_frame",
        "tiling",
        "bitrate_validation",
        "encoder",
    }
    assert not Path(args.output_dir).exists()


def test_tas_runner_command_matches_subprocess_adapter_interface(args) -> None:
    plan, _ = tas.build_plan(args)
    command = plan["commands"]["measured"][0]
    parsed = adapter_parser().parse_args(command[3:])
    assert parsed.engine == Path(args.engine)
    assert parsed.frames == 1000
    assert parsed.fps == "24/1"
    assert parsed.decode_method == "cpu"
    assert parsed.writer == "ffmpeg"
    paths = tas_suite.TasProcessPaths.for_output(parsed.output)
    assert parsed.encoder_contract == paths.encoder_contract
    assert parsed.lifecycle_output == paths.lifecycle
    assert parsed.runtime_evidence == paths.runtime_evidence


@pytest.mark.parametrize("decode_method", ["cpu", "nvdec"])
@pytest.mark.parametrize("writer", ["ffmpeg", "nelux"])
def test_tas_tuned_plan_accepts_all_four_io_paths(args, decode_method, writer) -> None:
    args.execution_profile = "tuned"
    args.decode_method = decode_method
    args.writer = writer
    args.frames = 300
    args.warmup_frames = 30
    args.runs = 1
    args.extra_runs = 0
    args.skip_bitrate_validation = True
    plan, _ = tas.build_plan(args)

    parameters = plan["parameters"]
    assert parameters["decode_method"] == decode_method
    assert parameters["writer"] == writer
    assert parameters["cuda_graph"] is True
    assert parameters["frames"] == 300
    assert parameters["warmup_frames"] == 30
    assert parameters["bitrate_validation"] is False
    assert parameters["max_compute_processes"] == (1 if writer == "nelux" else 2)


@pytest.mark.parametrize("overrides", [{"decode_method": "nvdec"}, {"writer": "nelux"}])
def test_tas_upstream_defaults_cannot_be_overridden(args, overrides) -> None:
    vars(args).update(overrides)
    with pytest.raises(CompetitorError, match="upstream-default requires"):
        tas.build_plan(args)


@pytest.mark.parametrize("decode_method,writer", [(None, None), ("cpu", None), (None, "nelux")])
def test_tas_tuned_requires_complete_io_profile(args, decode_method, writer) -> None:
    args.execution_profile = "tuned"
    args.decode_method = decode_method
    args.writer = writer
    with pytest.raises(CompetitorError, match="tuned requires explicit"):
        resolve_execution_profile(args)


def test_tas_upstream_metadata_must_match_executable_defaults(implementation) -> None:
    profile = TasExecutionProfile("upstream-default", "cpu", "ffmpeg")
    validate_declared_profile(implementation, profile)
    implementation["execution_profiles"]["upstream-default"]["cuda_graph"] = False
    with pytest.raises(CompetitorError, match="cuda_graph"):
        validate_declared_profile(implementation, profile)


def _evidence(**changes: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "valid",
        "errors": [],
        "decode_method": "nvdec",
        "writer": "nelux",
        "cuda_graph": True,
        "engine_reused": True,
        "processed_frames": 2,
        "engine_sha256": "e" * 64,
        "onnx_sha256": "c" * 64,
        "source_revision": REVISION,
        "adapter_sha256": "d" * 64,
        "runtime": RUNTIME,
        **changes,
    }


def _load_evidence(path: Path) -> dict[str, Any]:
    return tas_suite.load_runtime_evidence(
        path,
        profile=TasExecutionProfile("tuned", "nvdec", "nelux"),
        frames=2,
        engine_sha256="e" * 64,
        onnx_sha256="c" * 64,
        source_revision=REVISION,
        adapter_sha256="d" * 64,
        runtime_versions=RUNTIME,
    )


def test_tas_runtime_evidence_retains_actual_runtime_versions(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    write_json(path, _evidence())
    assert _load_evidence(path) == _evidence()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"schema_version": 2}, "schema"),
        ({"schema_version": True}, "schema"),
        ({"status": "invalid"}, "not valid"),
        ({"errors": ["fallback"]}, "not valid"),
        ({"errors": None}, "not valid"),
        ({"decode_method": "cpu"}, "decode_method"),
        ({"writer": "ffmpeg"}, "writer"),
        ({"cuda_graph": False}, "cuda_graph"),
        ({"engine_reused": False}, "engine_reused"),
        ({"engine_reused": 1}, "engine_reused"),
        ({"processed_frames": 1}, "processed_frames"),
        ({"engine_sha256": "wrong"}, "engine_sha256"),
        ({"onnx_sha256": "wrong"}, "onnx_sha256"),
        ({"source_revision": "wrong"}, "source_revision"),
        ({"adapter_sha256": "wrong"}, "adapter_sha256"),
        ({"runtime": None}, "version object"),
        ({"runtime": {**RUNTIME, "nelux": "unknown"}}, "nelux version"),
        ({"runtime": {**RUNTIME, "tensorrt": "different"}}, "tensorrt differs"),
    ],
)
def test_tas_runtime_evidence_rejects_fallback_rebuild_or_incomplete_provenance(
    tmp_path, changes, match
) -> None:
    path = tmp_path / "runtime.json"
    write_json(path, _evidence(**changes))
    with pytest.raises(CompetitorError, match=match):
        _load_evidence(path)


def test_tas_runtime_evidence_is_required(tmp_path) -> None:
    with pytest.raises(CompetitorError, match="Cannot read JSON"):
        _load_evidence(tmp_path / "missing.json")


def test_tas_command_waits_for_child_completion_and_preserves_logs(tmp_path) -> None:
    command = command_spec(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "print('adapter startup', file=sys.stderr, flush=True); "
            "subprocess.run([sys.executable, '-c', \"print('encoded')\"], check=True); "
            "print('adapter finalized', file=sys.stderr)",
        ]
    )
    stdout = tmp_path / "measured.stdout.log"
    stderr = tmp_path / "measured.stderr.log"
    result = tas_suite.run_tas_command(command, stdout, stderr)
    assert result.returncode == 0
    assert result.process_finished_ns > result.process_started_ns
    assert stdout.read_text() == "encoded\n"
    assert stderr.read_text() == "adapter startup\nadapter finalized\n"


def test_tas_command_reports_failed_start(tmp_path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    result = tas_suite.run_tas_command(command_spec([str(tmp_path / "missing")]), stdout, stderr)
    assert result.returncode == 127
    assert "Failed to start TAS adapter" in stderr.read_text()


@pytest.fixture
def suite(tmp_path, monkeypatch, implementation):
    events: list[str] = []
    active = False

    class Sampler:
        def __init__(self, _gpu_id, _sample_interval_ms):
            pass

        def initialize(self):
            return {}

        def shutdown(self):
            pass

        def start(self, _time):
            nonlocal active
            active = True
            events.append("start")

        def stop(self):
            nonlocal active
            active = False
            events.append("stop")
            return []

        def samples_relative_to(self, samples, _time):
            return samples

    monkeypatch.setattr(video_suite, "NvmlSampler", Sampler)
    monkeypatch.setattr(
        video_suite,
        "summarize_samples",
        lambda *_args, **_kwargs: {"valid": True, "errors": [], "power": {"limit_w": 350}},
    )
    monkeypatch.setattr(tas_suite, "collect_environment", lambda _gpu: {})
    monkeypatch.setattr(tas_suite, "adapter_sha256", lambda: "d" * 64)
    monkeypatch.setenv("TRTVIDEO_IMAGE_ID", "sha256:test")
    monkeypatch.setenv("TRTVIDEO_BUILD_REVISION", "clean-test-revision")
    monkeypatch.setenv("TRTVIDEO_BUILD_DIRTY", "0")
    assets = {}
    for kind in ("input", "onnx", "engine", "engine_manifest", "asset_lock", "benchmark_adapter"):
        path = tmp_path / kind
        path.write_bytes(kind.encode())
        assets[kind] = path
    profile = TasExecutionProfile("tuned", "nvdec", "nelux")
    encoder = NvencCbrContract(bitrate_bps=30_000_000, gop_frames=24).as_dict()
    config = tas_suite.TasSuiteConfig(
        implementation=implementation,
        workload_id="span-madrid",
        variant="720p",
        output_dir=tmp_path / "suite",
        frames=2,
        warmup_frames=1,
        output_contract={"width": 4, "height": 4, "fps": "24/1", "frames": 2},
        benchmark_contract={"contract_version": 2},
        assets=assets,
        policy=SuitePolicy(initial_runs=1, extra_runs=0, spread_threshold=0.05, idle_seconds=0),
        sample_interval_ms=100,
        gpu_id=0,
        profile=profile,
        implementation_parameters={
            **profile.as_parameters(),
            "batch_size": 1,
            "full_frame": True,
            "tiling": False,
            "bitrate_validation": True,
            "encoder": encoder,
        },
        runtime_versions=RUNTIME,
        command=lambda path, frames: command_spec(["tas-test", str(path), str(frames)]),
    )

    def validate(path, _contract):
        assert active is False
        events.append(f"validate:{path.name}")
        return {"valid": True, "errors": []}

    monkeypatch.setattr(tas_suite, "validate_output", validate)
    load_evidence = tas_suite.load_runtime_evidence

    def validate_evidence(*args, **kwargs):
        assert active is False
        return load_evidence(*args, **kwargs)

    monkeypatch.setattr(tas_suite, "load_runtime_evidence", validate_evidence)

    def install_process(*, invalid: str | None = None, lifecycle: bool = True) -> None:
        def execute(spec, _stdout, _stderr):
            output = Path(spec[0][1])
            frames = int(spec[0][2])
            events.append(f"execute:{output.name}")
            output.write_bytes(b"encoded-video")
            paths = tas_suite.TasProcessPaths.for_output(output)
            assert json.loads(paths.encoder_contract.read_text()) == encoder
            evidence = _evidence(
                processed_frames=frames,
                engine_sha256=sha256_file(assets["engine"]),
                onnx_sha256=sha256_file(assets["onnx"]),
            )
            if invalid == output.name:
                evidence["engine_reused"] = False
            write_json(paths.runtime_evidence, evidence)
            if lifecycle:
                write_frame_markers(
                    paths.lifecycle,
                    FrameLifecycleMarkers(
                        first_frame_completed_ns=1_100_000_100,
                        last_frame_completed_ns=1_900_000_100,
                        processed_frames=frames,
                        instrumentation="tas-output-queue-submission",
                    ),
                )
            return ProcessResult(0, 1_000_000_100, 3_000_000_100)

        monkeypatch.setattr(tas_suite, "run_tas_command", execute)

    install_process()
    return config, events, install_process


@pytest.mark.parametrize("keep_outputs", [False, True])
def test_tas_suite_uses_common_accounting_and_validates_runtime_outside_timing(
    suite, tmp_path, keep_outputs
) -> None:
    config, events, _install = suite
    summary, returncode = tas_suite.run_tas_suite(
        replace(config, keep_outputs=keep_outputs), root=tmp_path
    )
    run_dir = config.output_dir / "run-01"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert returncode == 0
    assert summary["status"] == "valid"
    assert summary["statistics"]["median_fps"] == 1.0
    assert events == [
        "execute:warmup.mp4",
        "validate:warmup.mp4",
        "start",
        "execute:output.mp4",
        "stop",
        "validate:output.mp4",
    ]
    assert manifest["measured"]["metrics"]["wall_time_sec"] == 2.0
    assert (
        manifest["measured"]["metrics"]["lifecycle"]["instrumentation"]
        == "tas-output-queue-submission"
    )
    assert manifest["measured"]["validation"]["runtime_evidence"]["runtime"] == RUNTIME
    assert manifest["warmup"]["validation"]["runtime_evidence"]["processed_frames"] == 1
    assert manifest["parameters"]["max_compute_processes"] == 1
    assert (run_dir / "output.mp4").exists() is keep_outputs
    assert (run_dir / "warmup.mp4").exists() is keep_outputs
    assert (run_dir / "output.runtime-evidence.json").is_file()


@pytest.mark.parametrize("invalid", ["warmup.mp4", "output.mp4"])
def test_tas_suite_rejects_successful_process_without_engine_reuse(
    suite, tmp_path, invalid
) -> None:
    config, events, install = suite
    install(invalid=invalid)
    summary, returncode = tas_suite.run_tas_suite(config, root=tmp_path)
    manifest = json.loads((config.output_dir / "run-01/manifest.json").read_text())
    assert returncode == 2
    assert summary["status"] == "invalid"
    assert any("engine_reused" in error for error in manifest["errors"])
    if invalid == "warmup.mp4":
        assert events == ["execute:warmup.mp4", "validate:warmup.mp4"]
    else:
        assert (config.output_dir / "run-01/output.mp4").is_file()


def test_tas_suite_requires_measured_lifecycle(suite, tmp_path) -> None:
    config, _, install = suite
    install(lifecycle=False)
    summary, returncode = tas_suite.run_tas_suite(config, root=tmp_path)
    manifest = json.loads((config.output_dir / "run-01/manifest.json").read_text())
    assert returncode == 2
    assert summary["status"] == "invalid"
    assert any("Lifecycle timing" in error for error in manifest["errors"])


def test_tas_suite_preserves_preexisting_evidence(suite, tmp_path) -> None:
    config, _, _ = suite
    config.output_dir.mkdir()
    old_evidence = config.output_dir / "suite.json"
    old_evidence.write_text("previous measurement")
    with pytest.raises(CompetitorError, match="output directory is not empty"):
        tas_suite.run_tas_suite(config, root=tmp_path)
    assert old_evidence.read_text() == "previous measurement"
