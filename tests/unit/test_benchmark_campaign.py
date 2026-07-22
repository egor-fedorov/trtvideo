from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_media.benchmarking.environment import sha256_file
from ai_media.video.nvenc import NvencCbrContract
from benchmarks.scripts.aggregate_campaign import (
    IMPLEMENTATIONS,
    CampaignError,
    aggregate_campaign,
)


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
