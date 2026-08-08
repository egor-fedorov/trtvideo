from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.campaign.aggregate import (
    IMPLEMENTATIONS,
    INFERENCE_PARITY_GAP,
    PREPROCESSING_DIAGNOSTIC_GAP,
    PRODUCT_OUTPUT_GAP,
    CampaignError,
    aggregate_campaign,
)
from benchmarks.scripts.campaign.core import (
    CONFIG_NAME,
    EVENT_LOG_NAME,
    CampaignConfig,
    CampaignEvent,
    CampaignEventError,
    append_event,
    campaign_steps,
    load_events,
    write_campaign_config,
)
from benchmarks.scripts.campaign.report import render_markdown
from benchmarks.scripts.campaign.run import CampaignRunError, run_campaign
from benchmarks.scripts.runtime.environment import sha256_file
from trtvideo.video.nvcodec.encoder import NvencCbrContract


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_removed_execution_profile_is_rejected() -> None:
    with pytest.raises(CampaignEventError, match="Unknown campaign"):
        CampaignConfig.create(
            execution_profile="parity",
            vstrt_arguments="",
            vsgan_arguments="",
        )


def _campaign(
    root: Path,
    fps: dict[str, list[float]],
    *,
    revision: str = "revision-1",
    contract_version: int = 1,
    measured_frames: int = 1000,
    warmup_frames: int = 100,
) -> Path:
    workload_path = root / "benchmarks/workload.json"
    _write_json(
        workload_path,
        {
            "benchmark": {
                "contract_version": contract_version,
                "warmup_frames": warmup_frames,
                "measured_frames": measured_frames,
                "initial_runs": 3,
                "extra_runs_on_spread": 2,
                "spread_threshold": 0.05,
                "idle_seconds": 10,
            },
            "quality": {
                "model_space": {
                    "contract_version": 3,
                    "frame_indices": [0, 499, 999],
                },
                "product_output": {
                    "frame_indices": [0, 499, 999],
                    "thresholds": {
                        "psnr_min_db": 35.0,
                        "ssim_min": 0.95,
                    },
                    "crops": [{"name": "center"}],
                },
            },
            "clip": {"frames": 1000},
        },
    )
    workload_sha = sha256_file(workload_path)
    campaign_dir = root / "artefacts/benchmarks/campaign"
    write_campaign_config(
        campaign_dir / CONFIG_NAME,
        CampaignConfig.create(
            execution_profile="upstream-default",
            vstrt_arguments="",
            vsgan_arguments="",
        ),
    )
    encoder = NvencCbrContract(
        bitrate_bps=60_000_000,
        gop_frames=24,
    ).as_dict()
    for implementation, product in IMPLEMENTATIONS.items():
        for round_index, value in enumerate(fps[implementation], start=1):
            engine_sha = "shared-engine" if implementation != "vsgan" else "vsgan-engine"
            wall_time_sec = measured_frames / value
            manifest: dict[str, Any] = {
                "status": "valid",
                "run_index": 1,
                "product": product,
                "workload_id": "workload-v1",
                "benchmark_contract_version": contract_version,
                "variant": "1080p",
                "parameters": {
                    "frames": measured_frames,
                    "warmup_frames": warmup_frames,
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
                        "wall_time_sec": wall_time_sec,
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
                        "lifecycle": {
                            "clock": "time.perf_counter_ns",
                            "boundary_contract": (
                                "process_start -> first_frame_completed -> "
                                "last_frame_completed -> process_exit"
                            ),
                            "instrumentation": f"{implementation}-instrumentation",
                            "startup_sec": wall_time_sec * 0.02,
                            "steady_state_frame_loop_sec": wall_time_sec * 0.95,
                            "finalize_mux_sec": wall_time_sec * 0.03,
                            "total_sec": wall_time_sec,
                            "processed_frames": measured_frames,
                            "steady_state_frames": measured_frames - 1,
                        },
                        "nvml": {
                            "utilization": {"average_gpu_percent": 90.0},
                            "power": {"average_w": 240.0, "joules_per_frame": 24.0},
                            "memory": {"peak_delta_mib": 4000.0},
                        },
                    },
                    "validation": {"observed": {"video_bitrate_bps": 60_000_000}},
                    "output": {"size_bytes": 300 * 1024 * 1024},
                },
            }
            if implementation == "vstrt":
                manifest["parameters"].update(
                    {
                        "execution_profile": "upstream-default",
                        "vspipe_requests": "auto",
                        "num_streams": 1,
                        "vapoursynth_threads": "auto",
                        "cuda_graph": False,
                    }
                )
            elif implementation == "vsgan":
                manifest["parameters"].update(
                    {
                        "execution_profile": "upstream-default",
                        "vspipe_requests": "auto",
                        "num_streams": 4,
                        "vapoursynth_threads": 4,
                        "cuda_graph": False,
                    }
                )
            _write_json(
                campaign_dir / implementation / f"round-{round_index:02d}" / "run-01/manifest.json",
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


def _tensor_report(root: Path, *, preprocessing: bool) -> Path:
    document_type = "preprocessing-diagnostic" if preprocessing else "inference-parity"
    status = "complete" if preprocessing else "valid"
    path = root / "artefacts/benchmarks/quality" / f"{document_type}.json"
    _write_json(
        path,
        {
            "document_type": document_type,
            "status": status,
            "publishable": True,
            "acceptance_gate": not preprocessing,
            "contract_version": 3,
            "workload_id": "workload-v1",
            "variant": "1080p",
            "execution_profile": "upstream-default",
            "frame_indices": [0, 499, 999],
            "assets": {
                "input_sha256": "input-sha",
                "onnx_sha256": "onnx-sha",
                **({"canonical_input_manifest_sha256": "a" * 64} if not preprocessing else {}),
            },
            "reference": {
                "implementation": "trtvideo",
                "capture_manifest_sha256": "a" * 64,
                "engine_sha256": "shared-engine",
                "execution_profile": {
                    "execution_profile": "upstream-default",
                    "cuda_graph": False,
                },
                "image": {
                    "id": "trtvideo-image",
                    "repository_revision": "revision-1",
                    "source_dirty": "0",
                },
            },
            "comparisons": [
                {
                    "implementation": "vs-mlrt",
                    "status": status,
                    "capture_manifest_sha256": "b" * 64,
                    **({"canonical_input_manifest_sha256": "a" * 64} if not preprocessing else {}),
                    "engine_sha256": "shared-engine",
                    "execution_profile": {
                        "execution_profile": "upstream-default",
                        "vspipe_requests": "auto",
                        "num_streams": 1,
                        "vapoursynth_threads": "auto",
                        "cuda_graph": False,
                    },
                    "image": {
                        "id": "vstrt-image",
                        "repository_revision": "revision-1",
                        "source_dirty": "0",
                    },
                },
                {
                    "implementation": "VSGAN-tensorrt-docker",
                    "status": status,
                    "capture_manifest_sha256": "c" * 64,
                    **({"canonical_input_manifest_sha256": "a" * 64} if not preprocessing else {}),
                    "engine_sha256": "vsgan-engine",
                    "execution_profile": {
                        "execution_profile": "upstream-default",
                        "vspipe_requests": "auto",
                        "num_streams": 4,
                        "vapoursynth_threads": 4,
                        "cuda_graph": False,
                    },
                    "image": {
                        "id": "vsgan-image",
                        "repository_revision": "revision-1",
                        "source_dirty": "0",
                    },
                },
            ],
        },
    )
    return path


def _inference_report(root: Path) -> Path:
    return _tensor_report(root, preprocessing=False)


def _preprocessing_report(root: Path) -> Path:
    return _tensor_report(root, preprocessing=True)


def _product_output_report(root: Path, *, contract_version: int = 1) -> Path:
    report_dir = root / "artefacts/benchmarks/quality/product-output"
    report_dir.mkdir(parents=True, exist_ok=True)

    def artifact(name: str, content: bytes) -> tuple[str, str]:
        path = report_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path.relative_to(root).as_posix(), sha256_file(path)

    encoder = NvencCbrContract(
        bitrate_bps=60_000_000,
        gop_frames=24,
    ).as_dict()

    def run_manifest(
        name: str,
        *,
        product: str,
        implementation: str,
        engine_sha256: str,
    ) -> tuple[str, str]:
        path = report_dir / name
        parameters = {
            "frames": 1000,
            "encoder": encoder,
        }
        if implementation == "vstrt":
            parameters.update(
                {
                    "execution_profile": "upstream-default",
                    "vspipe_requests": "auto",
                    "num_streams": 1,
                    "vapoursynth_threads": "auto",
                    "cuda_graph": False,
                }
            )
        elif implementation == "vsgan":
            parameters.update(
                {
                    "execution_profile": "upstream-default",
                    "vspipe_requests": "auto",
                    "num_streams": 4,
                    "vapoursynth_threads": 4,
                    "cuda_graph": False,
                }
            )
        manifest = {
            "status": "valid",
            "product": product,
            "workload_id": "workload-v1",
            "benchmark_contract_version": contract_version,
            "variant": "1080p",
            "parameters": parameters,
            "assets": {
                "input": {"sha256": "input-sha"},
                "onnx": {"sha256": "onnx-sha"},
                "engine": {"sha256": engine_sha256},
            },
            "environment": {
                "image": {
                    "id": f"{implementation}-image",
                    "repository_revision": "revision-1",
                    "source_dirty": "0",
                }
            },
            "reproducibility": {"publishable": True},
            "measured": {"validation": {"valid": True}},
        }
        _write_json(
            path,
            manifest,
        )
        return path.relative_to(root).as_posix(), sha256_file(path)

    reference_manifest, reference_manifest_sha = run_manifest(
        "trtvideo/run-01/manifest.json",
        product="trtvideo",
        implementation="trtvideo",
        engine_sha256="shared-engine",
    )
    comparisons = []
    for product, implementation, engine in (
        ("vs-mlrt", "vstrt", "shared-engine"),
        ("VSGAN-tensorrt-docker", "vsgan", "vsgan-engine"),
    ):
        manifest_path, run_manifest_sha = run_manifest(
            f"{implementation}/run-01/manifest.json",
            product=product,
            implementation=implementation,
            engine_sha256=engine,
        )
        metrics = {}
        for metric in ("psnr", "ssim"):
            stats_path, stats_sha = artifact(
                f"{implementation}/{metric}.stats.log",
                f"{metric} stats".encode(),
            )
            log_path, log_sha = artifact(
                f"{implementation}/{metric}.ffmpeg.log",
                f"{metric} log".encode(),
            )
            metrics[metric] = {
                "stats_path": stats_path,
                "stats_sha256": stats_sha,
                "ffmpeg_log": log_path,
                "ffmpeg_log_sha256": log_sha,
            }
        comparisons.append(
            {
                "implementation": product,
                "status": "valid",
                "engine_sha256": engine,
                "run_manifest": manifest_path,
                "run_manifest_sha256": run_manifest_sha,
                "metrics": metrics,
            }
        )

    visual_crops = {}
    for implementation, directory in (
        ("trtvideo", "trtvideo"),
        ("vs-mlrt", "vstrt"),
        ("VSGAN-tensorrt-docker", "vsgan"),
    ):
        crops = []
        for frame_index in (0, 499, 999):
            crop_path, checksum = artifact(
                f"crops/{directory}/frame-{frame_index:06d}.center.png",
                f"{implementation} {frame_index}".encode(),
            )
            crops.append(
                {
                    "frame_index": frame_index,
                    "crop": "center",
                    "path": crop_path,
                    "sha256": checksum,
                }
            )
        visual_crops[implementation] = crops

    report_path = report_dir / "product-output-parity.json"
    _write_json(
        report_path,
        {
            "document_type": "product-output-parity",
            "status": "valid",
            "publishable": True,
            "workload_id": "workload-v1",
            "variant": "1080p",
            "frame_indices": [0, 499, 999],
            "thresholds": {
                "psnr_min_db": 35.0,
                "ssim_min": 0.95,
            },
            "assets": {
                "input_sha256": "input-sha",
                "onnx_sha256": "onnx-sha",
            },
            "reference": {
                "implementation": "trtvideo",
                "engine_sha256": "shared-engine",
                "run_manifest": reference_manifest,
                "run_manifest_sha256": reference_manifest_sha,
            },
            "comparisons": comparisons,
            "visual_crops": visual_crops,
        },
    )
    return report_path


def test_aggregate_campaign_builds_acceptance_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "valid"
    assert summary["benchmark_contract_version"] == 1
    assert summary["publishable"] is False
    assert summary["publication"]["errors"] == [
        INFERENCE_PARITY_GAP,
        PREPROCESSING_DIAGNOSTIC_GAP,
        PRODUCT_OUTPUT_GAP,
    ]
    assert summary["needs_extra_runs"] is False
    assert summary["execution_profile"] == "upstream-default"
    assert summary["parameters"]["rounds"] == 3
    assert summary["parameters"]["execution_profiles"]["vstrt"] == {
        "execution_profile": "upstream-default",
        "vspipe_requests": "auto",
        "num_streams": 1,
        "vapoursynth_threads": "auto",
        "cuda_graph": False,
    }
    assert summary["implementations"]["trtvideo"]["statistics"]["median_fps"] == 10.0
    assert summary["implementations"]["trtvideo"]["statistics"]["median_cpu_cores"] == 1.0
    assert summary["implementations"]["trtvideo"]["statistics"]["median_startup_sec"] == 2.0
    assert summary["implementations"]["vstrt"]["relative_to_trtvideo_percent"] == pytest.approx(
        -10.0
    )


def test_aggregate_campaign_accepts_matching_inference_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(
        campaign_dir,
        root=tmp_path,
        idle_seconds=10,
        inference_report=_inference_report(tmp_path),
    )

    assert summary["publication"]["errors"] == [
        PREPROCESSING_DIAGNOSTIC_GAP,
        PRODUCT_OUTPUT_GAP,
    ]
    assert summary["quality"]["inference_parity"]["status"] == "valid"


def test_aggregate_campaign_is_publishable_with_all_quality_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(
        campaign_dir,
        root=tmp_path,
        idle_seconds=10,
        inference_report=_inference_report(tmp_path),
        preprocessing_report=_preprocessing_report(tmp_path),
        product_output_report=_product_output_report(tmp_path),
    )

    assert summary["publishable"] is True
    assert summary["publication"] == {
        "ready": True,
        "errors": [],
        "warnings": [],
    }
    assert summary["quality"]["product_output"]["status"] == "valid"


def test_product_output_quality_keeps_full_clip_for_shorter_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
        contract_version=2,
        measured_frames=400,
        warmup_frames=30,
    )

    summary = aggregate_campaign(
        campaign_dir,
        root=tmp_path,
        idle_seconds=10,
        inference_report=_inference_report(tmp_path),
        preprocessing_report=_preprocessing_report(tmp_path),
        product_output_report=_product_output_report(
            tmp_path,
            contract_version=2,
        ),
    )

    assert summary["publishable"] is True
    assert summary["benchmark_contract_version"] == 2
    assert summary["parameters"]["measured_frames"] == 400
    assert summary["quality"]["product_output"]["status"] == "valid"


def test_aggregate_campaign_rejects_inference_engine_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    report_path = _inference_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparisons"][1]["engine_sha256"] = "other-engine"
    _write_json(report_path, report)

    with pytest.raises(CampaignError, match="VSGAN-tensorrt-docker engine"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            inference_report=report_path,
        )


def test_aggregate_campaign_rejects_inference_profile_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    report_path = _inference_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparisons"][0]["execution_profile"]["num_streams"] = 2
    _write_json(report_path, report)

    with pytest.raises(CampaignError, match="vs-mlrt execution profile"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            inference_report=report_path,
        )


def test_aggregate_campaign_rejects_inference_image_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    report_path = _inference_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparisons"][0]["image"]["id"] = "other-image"
    _write_json(report_path, report)

    with pytest.raises(CampaignError, match="vs-mlrt image"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            inference_report=report_path,
        )


def test_aggregate_campaign_rejects_product_output_image_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    report_path = _product_output_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / report["comparisons"][0]["run_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["image"]["id"] = "other-image"
    _write_json(manifest_path, manifest)
    report["comparisons"][0]["run_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(report_path, report)

    with pytest.raises(CampaignError, match="vs-mlrt run changed image"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            product_output_report=report_path,
        )


def test_aggregate_campaign_rejects_product_output_profile_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    report_path = _product_output_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / report["comparisons"][0]["run_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"]["num_streams"] = 2
    _write_json(manifest_path, manifest)
    report["comparisons"][0]["run_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(report_path, report)

    with pytest.raises(CampaignError, match="changed execution profile"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            product_output_report=report_path,
        )


def test_aggregate_campaign_requests_two_extra_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 12.0, 8.0],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "needs-extra-runs"
    assert summary["needs_extra_runs"] is True
    assert summary["unstable_implementations"] == ["trtvideo"]


def test_aggregate_campaign_rejects_mixed_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
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


def test_aggregate_campaign_rejects_mixed_benchmark_contract_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    path = campaign_dir / "vstrt/round-02/run-01/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["benchmark_contract_version"] = 2
    _write_json(path, manifest)

    with pytest.raises(CampaignError, match="benchmark contract version"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_rejects_mixed_execution_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )
    path = campaign_dir / "vstrt/round-02/run-01/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["parameters"]["num_streams"] = 2
    _write_json(path, manifest)

    with pytest.raises(CampaignError, match="execution profile changed"):
        aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)


def test_aggregate_campaign_rejects_requested_profile_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
            "vstrt": [9.0, 9.1, 8.9],
            "vsgan": [8.8, 8.9, 8.7],
        },
    )

    with pytest.raises(CampaignError, match="does not match aggregation request"):
        aggregate_campaign(
            campaign_dir,
            root=tmp_path,
            idle_seconds=10,
            execution_profile="tuned",
        )


def test_aggregate_campaign_rejects_different_cpu_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
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
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
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
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
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
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9],
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
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9, 10.0, 10.1],
            "vstrt": [9.0, 9.1, 8.9, 9.0, 9.1],
            "vsgan": [8.8, 8.9, 8.7, 8.8, 8.9],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    assert summary["status"] == "valid"
    assert summary["parameters"]["rounds"] == 5
    assert summary["rounds"][3]["order"] == ["vsgan", "vstrt", "trtvideo"]


def test_aggregate_campaign_accepts_four_of_five_consensus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    vstrt_fps = [
        9.206078920416692,
        9.776448783606835,
        9.418761837122538,
        9.348216376211594,
        9.215761747924864,
    ]
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [25.1, 25.0, 24.9, 25.1, 25.0],
            "vstrt": vstrt_fps,
            "vsgan": [9.3, 9.2, 9.1, 9.3, 9.2],
        },
    )

    summary = aggregate_campaign(
        campaign_dir,
        root=tmp_path,
        idle_seconds=10,
        inference_report=_inference_report(tmp_path),
        preprocessing_report=_preprocessing_report(tmp_path),
        product_output_report=_product_output_report(tmp_path),
    )

    result = summary["implementations"]["vstrt"]
    stability = result["stability"]
    assert summary["status"] == "valid"
    assert summary["publishable"] is True
    assert summary["unstable_implementations"] == []
    assert summary["stable_with_outlier_implementations"] == ["vstrt"]
    assert result["statistics"]["values_fps"] == vstrt_fps
    assert result["statistics"]["median_fps"] == vstrt_fps[3]
    assert stability["status"] == "stable-with-one-outlier"
    assert stability["full_relative_spread"] == pytest.approx(0.0610137635)
    assert stability["consensus"]["rounds"] == [1, 3, 4, 5]
    assert stability["consensus"]["relative_spread"] == pytest.approx(0.0229, abs=1e-4)
    assert stability["outlier"] == {
        "round": 2,
        "fps": vstrt_fps[1],
    }
    assert len(summary["publication"]["warnings"]) == 1
    markdown = render_markdown(summary)
    assert "stable-with-one-outlier" in markdown
    assert "round 2: 9.776 FPS" in markdown


def test_aggregate_campaign_rejects_five_runs_without_consensus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTVIDEO_BUILD_REVISION", raising=False)
    campaign_dir = _campaign(
        tmp_path,
        {
            "trtvideo": [10.0, 10.1, 9.9, 10.0, 10.1],
            "vstrt": [8.0, 9.0, 10.0, 11.0, 12.0],
            "vsgan": [8.8, 8.9, 8.7, 8.8, 8.9],
        },
    )

    summary = aggregate_campaign(campaign_dir, root=tmp_path, idle_seconds=10)

    stability = summary["implementations"]["vstrt"]["stability"]
    assert summary["status"] == "unstable"
    assert summary["publishable"] is False
    assert summary["unstable_implementations"] == ["vstrt"]
    assert summary["stable_with_outlier_implementations"] == []
    assert stability["status"] == "unstable"
    assert stability["consensus"]["accepted"] is False
    assert stability["outlier"] is None


def test_campaign_coordinator_records_actual_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign_dir = tmp_path / "artefacts/benchmarks/campaign"
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()

    def fake_make(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check is False
        assert "EXECUTION_PROFILE=upstream-default" in command
        target = command[3]
        if target == "aggregate-campaign":
            _write_json(campaign_dir / "campaign.json", {"status": "valid"})
            return subprocess.CompletedProcess(command, 0)
        implementation = target.removeprefix("campaign-")
        round_index = int(command[4].split("=", 1)[1])
        path = campaign_dir / implementation / f"round-{round_index:02d}" / "run-01/manifest.json"
        _write_json(path, {"status": "valid"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "benchmarks.scripts.campaign.run.subprocess.run",
        fake_make,
    )
    args = argparse.Namespace(
        campaign_dir=str(campaign_dir),
        make_campaign_dir="artefacts/benchmarks/campaign",
        benchmarks_dir=str(benchmarks_dir),
        idle_seconds=0.0,
        make_command="make",
        execution_profile="upstream-default",
        vstrt_arguments="",
        vsgan_arguments="",
        resume=False,
    )

    assert run_campaign(args) == 0
    events = load_events(campaign_dir / EVENT_LOG_NAME)

    assert [event.implementation for event in events] == [
        step.implementation for step in campaign_steps(3)
    ]
    assert all(event.status == "completed" for event in events)
    config = json.loads((campaign_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["execution_profile"] == "upstream-default"
    output = capsys.readouterr().out
    assert "[campaign 1/9] trtvideo, round 1/3" in output
    assert "[campaign 9/9] vstrt, round 3/3" in output


def test_campaign_coordinator_runs_extra_rounds_from_aggregate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir = tmp_path / "artefacts/benchmarks/campaign"
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    aggregate_calls = 0

    def fake_make(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        nonlocal aggregate_calls
        assert check is False
        target = command[3]
        if target == "aggregate-campaign":
            aggregate_calls += 1
            status = "needs-extra-runs" if aggregate_calls == 1 else "valid"
            _write_json(campaign_dir / "campaign.json", {"status": status})
            return subprocess.CompletedProcess(command, 0)
        implementation = target.removeprefix("campaign-")
        round_index = int(command[4].split("=", 1)[1])
        path = campaign_dir / implementation / f"round-{round_index:02d}" / "run-01/manifest.json"
        _write_json(path, {"status": "valid"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "benchmarks.scripts.campaign.run.subprocess.run",
        fake_make,
    )
    args = argparse.Namespace(
        campaign_dir=str(campaign_dir),
        make_campaign_dir="artefacts/benchmarks/campaign",
        benchmarks_dir=str(benchmarks_dir),
        idle_seconds=0.0,
        make_command="make",
        execution_profile="upstream-default",
        vstrt_arguments="",
        vsgan_arguments="",
        resume=False,
    )

    assert run_campaign(args) == 0
    events = load_events(campaign_dir / EVENT_LOG_NAME)

    assert aggregate_calls == 2
    assert len(events) == len(campaign_steps(5))
    assert events[-1].round_index == 5


def test_campaign_coordinator_rejects_changed_profile_arguments_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir = tmp_path / "artefacts/benchmarks/campaign"
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    write_campaign_config(
        campaign_dir / CONFIG_NAME,
        CampaignConfig.create(
            execution_profile="tuned",
            vstrt_arguments="--requests auto --num-streams 2",
            vsgan_arguments="--requests auto --num-streams 4",
        ),
    )
    monkeypatch.setattr(
        "benchmarks.scripts.campaign.run.subprocess.run",
        lambda command, check: subprocess.CompletedProcess(command, 0),
    )
    args = argparse.Namespace(
        campaign_dir=str(campaign_dir),
        make_campaign_dir="artefacts/benchmarks/campaign",
        benchmarks_dir=str(benchmarks_dir),
        idle_seconds=0.0,
        make_command="make",
        execution_profile="tuned",
        vstrt_arguments="--requests auto --num-streams 3",
        vsgan_arguments="--requests auto --num-streams 4",
        resume=True,
    )

    with pytest.raises(CampaignRunError, match="runner arguments changed"):
        run_campaign(args)
