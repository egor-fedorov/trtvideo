from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.scripts.runtime import video_suite
from benchmarks.scripts.runtime.cpu import ChildCpuSnapshot, ChildCpuUsage
from benchmarks.scripts.runtime.video_suite import (
    ProcessInvocation,
    ProcessResult,
    VideoRunPaths,
    VideoRunSpec,
    run_video_measurement,
)
from trtvideo.benchmarking.lifecycle import FrameLifecycleMarkers
from trtvideo.benchmarking.validation import OutputContract


class FakeSampler:
    def start(self, _start_time: float) -> None:
        pass

    def stop(self) -> list[Any]:
        return []

    def samples_relative_to(
        self,
        samples: list[Any],
        _start_time: float,
    ) -> list[Any]:
        return samples


def _contract(frames: int) -> OutputContract:
    return OutputContract(
        width=4,
        height=4,
        fps="24/1",
        frames=frames,
    )


def _manifest_fields() -> dict[str, Any]:
    return {
        "product": "test-product",
        "workload_id": "test-workload",
        "benchmark_contract_version": 1,
        "variant": "test",
        "parameters": {},
        "assets": {},
        "environment": {
            "image": {
                "id": "sha256:test",
                "repository_revision": "revision",
                "source_dirty": "0",
            }
        },
    }


def _patch_accounting(monkeypatch) -> None:
    snapshots = iter(
        (
            ChildCpuSnapshot(1.0, 2.0),
            ChildCpuSnapshot(1.1, 2.1),
        )
    )
    monkeypatch.setattr(video_suite, "snapshot_child_cpu", lambda: next(snapshots))
    monkeypatch.setattr(
        video_suite,
        "summarize_child_cpu",
        lambda *_args, **_kwargs: ChildCpuUsage(
            user_time_sec=0.1,
            system_time_sec=0.1,
            total_time_sec=0.2,
            average_cores=0.2,
            available_logical_cpus=1,
            capacity_percent=20.0,
        ),
    )
    monkeypatch.setattr(
        video_suite,
        "summarize_samples",
        lambda *_args, **_kwargs: {
            "valid": True,
            "errors": [],
            "power": {"limit_w": 250.0},
        },
    )


def test_shared_video_run_owns_measurement_and_manifest_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_accounting(monkeypatch)
    paths = VideoRunPaths.create(tmp_path / "results", 1)

    def execute(output: Path, start_ns: int, end_ns: int):
        def run(_stdout: Path, _stderr: Path) -> ProcessResult:
            output.write_bytes(b"video")
            return ProcessResult(0, start_ns, end_ns)

        return run

    manifest = run_video_measurement(
        VideoRunSpec(
            run_index=1,
            frames=2,
            warmup_frames=1,
            keep_outputs=True,
            max_compute_processes=1,
            max_graphics_processes=0,
            require_reproducible_environment=True,
            manifest_fields=_manifest_fields(),
            warmup=ProcessInvocation(
                command=["warmup"],
                execute=execute(paths.warmup_output, 10, 20),
            ),
            measured=ProcessInvocation(
                command=["measured"],
                execute=execute(paths.measured_output, 100, 1_000_000_100),
            ),
            warmup_contract=_contract(1),
            measured_contract=_contract(2),
            lifecycle_reader=lambda _result: FrameLifecycleMarkers(
                first_frame_completed_ns=100_000_100,
                last_frame_completed_ns=900_000_100,
                processed_frames=2,
                instrumentation="test",
            ),
        ),
        paths=paths,
        sampler=FakeSampler(),  # type: ignore[arg-type]
        root=tmp_path,
        validate=lambda _path, _contract: {"valid": True, "errors": []},
    )

    persisted = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest == persisted
    assert manifest["status"] == "valid"
    assert manifest["commands"] == {"warmup": ["warmup"], "measured": ["measured"]}
    assert manifest["measured"]["metrics"]["wall_time_sec"] == 1.0
    assert manifest["measured"]["metrics"]["end_to_end_fps"] == 2.0
    assert manifest["measured"]["metrics"]["cpu"]["average_cores"] == 0.2
    assert manifest["measured"]["metrics"]["lifecycle"]["processed_frames"] == 2
    assert manifest["measured"]["output"]["sha256"]


def test_invalid_warmup_does_not_execute_measured_process(tmp_path: Path) -> None:
    paths = VideoRunPaths.create(tmp_path / "results", 1)
    measured_called = False

    def fail_warmup(_stdout: Path, _stderr: Path) -> ProcessResult:
        return ProcessResult(1, 10, 20)

    def measure(_stdout: Path, _stderr: Path) -> ProcessResult:
        nonlocal measured_called
        measured_called = True
        return ProcessResult(0, 20, 30)

    manifest = run_video_measurement(
        VideoRunSpec(
            run_index=1,
            frames=2,
            warmup_frames=1,
            keep_outputs=False,
            max_compute_processes=1,
            max_graphics_processes=0,
            require_reproducible_environment=True,
            manifest_fields=_manifest_fields(),
            warmup=ProcessInvocation(command=["warmup"], execute=fail_warmup),
            measured=ProcessInvocation(command=["measured"], execute=measure),
            warmup_contract=_contract(1),
            measured_contract=_contract(2),
            lifecycle_reader=lambda _result: FrameLifecycleMarkers(1, 2, 2, "test"),
        ),
        paths=paths,
        sampler=FakeSampler(),  # type: ignore[arg-type]
        root=tmp_path,
        validate=lambda _path, _contract: {"valid": True, "errors": []},
    )

    assert manifest["status"] == "invalid"
    assert manifest["errors"] == [
        "Warmup process exited with code 1",
        "Warmup output was not created",
    ]
    assert measured_called is False
