from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_media.benchmarking.environment import sha256_file
from ai_media.video.nvenc import NvencCbrContract
from benchmarks.scripts.aggregate_campaign import (
    IMPLEMENTATIONS,
    CampaignError,
    aggregate_campaign,
)
from benchmarks.scripts.campaign import (
    EVENT_LOG_NAME,
    CampaignEvent,
    append_event,
    campaign_steps,
    load_events,
)
from benchmarks.scripts.run_campaign import run_campaign


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign(
    root: Path,
    fps: dict[str, list[float]],
    *,
    revision: str = "revision-1",
) -> Path:
    workload_path = root / "benchmarks/workload.json"
    _write_json(
        workload_path,
        {
            "benchmark": {
                "warmup_frames": 100,
                "measured_frames": 1000,
                "initial_runs": 3,
                "extra_runs_on_spread": 2,
                "spread_threshold": 0.05,
                "idle_seconds": 10,
            }
        },
    )
    workload_sha = sha256_file(workload_path)
    campaign_dir = root / "artefacts/benchmarks/campaign"
    encoder = NvencCbrContract(
        bitrate_bps=60_000_000,
        gop_frames=24,
    ).as_dict()
    for implementation, product in IMPLEMENTATIONS.items():
        for round_index, value in enumerate(fps[implementation], start=1):
            engine_sha = "shared-engine" if implementation != "vsgan" else "vsgan-engine"
            manifest = {
                "status": "valid",
                "run_index": 1,
                "product": product,
                "workload_id": "workload-v1",
                "variant": "1080p",
                "parameters": {
                    "frames": 1000,
                    "warmup_frames": 100,
                    "encoder": encoder,
                },
                "assets": {
                    "input": {"sha256": "input-sha"},
                    "onnx": {"sha256": "onnx-sha"},
                    "engine": {"sha256": engine_sha},
                    "workload_manifest": {
                        "path": "benchmarks/workload.json",
                        "sha256": workload_sha,
                    },
                },
                "environment": {
                    "gpu": {"name": "RTX 3090", "power_limit_w": 250.0},
                    "cpu": {"model": "CPU", "logical_cores": 16},
                    "image": {
                        "id": f"{implementation}-image",
                        "repository_revision": revision,
                        "source_dirty": "0",
                    },
                },
                "reproducibility": {"publishable": True, "errors": []},
                "measured": {
                    "metrics": {
                        "end_to_end_fps": value,
                        "wall_time_sec": 1000 / value,
                        "cpu": {
                            "accounting": "getrusage(RUSAGE_CHILDREN)",
                            "scope": "measured-child-process-tree",
                            "user_time_sec": 90.0,
                            "system_time_sec": 10.0,
                            "total_time_sec": 100.0,
                            "average_cores": 1.0,
                            "available_logical_cpus": 16,
                            "capacity_percent": 6.25,
                        },
                        "nvml": {
                            "utilization": {"average_gpu_percent": 90.0},
                            "power": {"average_w": 240.0, "joules_per_frame": 24.0},
                            "memory": {"peak_delta_mib": 4000.0},
                        },
                    },
                    "validation": {
                        "observed": {"video_bitrate_bps": 60_000_000}
                    },
                    "output": {"size_bytes": 300 * 1024 * 1024},
                },
            }
            _write_json(
                campaign_dir
                / implementation
                / f"round-{round_index:02d}"
                / "run-01/manifest.json",
                manifest,
            )
    rounds = len(next(iter(fps.values())))
    previous_finished: datetime | None = None
    events_path = campaign_dir / EVENT_LOG_NAME
    for attempt_index, step in enumerate(campaign_steps(rounds), start=1):
        started = (
            datetime(2026, 1, 1, tzinfo=UTC)
            if previous_finished is None
            else previous_finished + timedelta(seconds=10)
        )
        finished = started + timedelta(seconds=100)
        append_event(
            events_path,
            CampaignEvent(
                schema_version=1,
                attempt_index=attempt_index,
                sequence_index=step.sequence_index,
                round_index=step.round_index,
                implementation=step.implementation,
                status="completed",
                returncode=0,
                required_idle_seconds=0 if previous_finished is None else 10,
                observed_idle_seconds=0 if previous_finished is None else 10,
                started_at_utc=started.isoformat(),
                finished_at_utc=finished.isoformat(),
                duration_seconds=100,
                manifest=step.manifest_path.as_posix(),
            ),
        )
        previous_finished = finished
    return campaign_dir


def test_aggregate_campaign_builds_acceptance_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "valid"
    assert summary["publishable"] is False
    assert summary["needs_extra_runs"] is False
    assert summary["parameters"]["rounds"] == 3
    assert summary["implementations"]["ai-media"]["statistics"]["median_fps"] == 10.0
    assert summary["implementations"]["ai-media"]["statistics"]["median_cpu_cores"] == 1.0
    assert summary["implementations"]["vstrt"][
        "relative_to_ai_media_percent"
    ] == pytest.approx(-10.0)


def test_aggregate_campaign_requests_two_extra_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 12.0, 8.0],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "needs-extra-runs"
    assert summary["needs_extra_runs"] is True
    assert summary["unstable_implementations"] == ["ai-media"]


def test_aggregate_campaign_rejects_mixed_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    path = campaign_dir / "vsgan/round-02/run-01/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["environment"]["image"]["repository_revision"] = "other-revision"
    _write_json(path, manifest)

    with pytest.raises(CampaignError, match="repository revision"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_rejects_different_cpu_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    path = campaign_dir / "vsgan/round-02/run-01/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["measured"]["metrics"]["cpu"]["available_logical_cpus"] = 8
    _write_json(path, manifest)

    with pytest.raises(CampaignError, match="CPU accounting"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_requires_execution_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    (campaign_dir / EVENT_LOG_NAME).unlink()

    with pytest.raises(CampaignError, match="execution log"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_rejects_unobserved_idle_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    events_path = campaign_dir / EVENT_LOG_NAME
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[1]["observed_idle_seconds"] = 0
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="idle interval"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_rejects_declared_order_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    events_path = campaign_dir / EVENT_LOG_NAME
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[1]["implementation"] = "vsgan"
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="implementation"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_accepts_complete_five_round_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_MEDIA_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "ai-media": [10.0, 10.1, 9.9, 10.0, 10.1],
            "vstrt": [9.0, 9.1, 8.9, 9.0, 9.1],
            "vsgan": [8.8, 8.9, 8.7, 8.8, 8.9],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "valid"
    assert summary["parameters"]["rounds"] == 5
    assert summary["rounds"][3]["order"] == ["vsgan", "vstrt", "ai-media"]


def test_campaign_coordinator_records_actual_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir = tmp_path / "artefacts/benchmarks/campaign"
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()

    def fake_make(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check is False
        target = command[3]
        if target == "aggregate-campaign":
            return subprocess.CompletedProcess(command, 0)
        implementation = target.removeprefix("campaign-")
        round_index = int(command[4].split("=", 1)[1])
        path = (
            campaign_dir
            / implementation
            / f"round-{round_index:02d}"
            / "run-01/manifest.json"
        )
        _write_json(path, {"status": "valid"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "benchmarks.scripts.run_campaign.subprocess.run",
        fake_make,
    )
    args = argparse.Namespace(
        campaign_dir=str(campaign_dir),
        make_campaign_dir="artefacts/benchmarks/campaign",
        benchmarks_dir=str(benchmarks_dir),
        idle_seconds=0.0,
        make_command="make",
        resume=False,
    )

    assert run_campaign(args) == 0
    events = load_events(campaign_dir / EVENT_LOG_NAME)

    assert [event.implementation for event in events] == [
        step.implementation for step in campaign_steps(3)
    ]
    assert all(event.status == "completed" for event in events)
