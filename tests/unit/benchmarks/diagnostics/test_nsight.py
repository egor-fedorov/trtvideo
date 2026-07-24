from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scripts.diagnostics.nsight import (
    OPTIONAL_EMPTY_STATS_REPORTS,
    STATS_REPORTS,
    TRACE_APIS,
    NsightPaths,
    build_nsight_command,
    build_plan,
    build_stats_command,
    build_upscale_command,
    gpu_video_preflight_error,
)


def manifest() -> dict:
    return {
        "id": "span-test",
        "clip": {
            "variants": [
                {
                    "name": "1080p",
                    "path": "videos/span_1080p.mp4",
                    "benchmark_output": {
                        "width": 3840,
                        "height": 2160,
                        "bitrate_mbps": 60,
                    },
                }
            ]
        },
    }


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        variant="1080p",
        engine="models/span_1080p.engine",
        output_dir=str(tmp_path / "diagnostic"),
        gpu_id=0,
        frames=120,
        nsys="nsys",
        json=None,
        dry_run=False,
    )


def test_upscale_command_uses_regular_nvcodec_path(tmp_path: Path) -> None:
    command = build_upscale_command(
        args(tmp_path),
        manifest(),
        output_path=tmp_path / "output.mp4",
    )

    assert command[:3] == ["upscale", "--backend", "nvcodec"]
    assert command[command.index("--max-frames") + 1] == "120"
    assert command[command.index("--bitrate-mbps") + 1] == "60"
    assert "--profile" not in command
    assert "--profile-json" not in command
    assert "--cuda-graph" not in command


def test_nsight_command_enables_required_trace_providers(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    paths = NsightPaths.create(Path(parsed.output_dir))

    command = build_nsight_command(parsed, manifest(), paths=paths)

    assert command[:3] == ["nsys", "profile", f"--trace={TRACE_APIS}"]
    assert "--sample=none" in command
    assert "--cpuctxsw=none" in command
    assert "--gpu-video-devices=0" in command
    assert command[command.index("--output") + 1] == str(paths.trace_base)
    assert command[command.index("upscale") :] == build_upscale_command(
        parsed,
        manifest(),
        output_path=paths.video,
    )


def test_stats_command_refreshes_sqlite_only_when_requested(tmp_path: Path) -> None:
    command = build_stats_command(
        "nsys",
        STATS_REPORTS[0],
        tmp_path / "trace.nsys-rep",
        force_export=True,
    )

    assert "--force-export=true" in command
    assert command[command.index("--report") + 1] == STATS_REPORTS[0]
    assert command[command.index("--format") + 1] == "csv"


def test_dry_run_plan_is_explicitly_non_publishable(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    Path(parsed.manifest).write_text(
        """{
  "id": "span-test",
  "clip": {
    "variants": [
      {
        "name": "1080p",
        "path": "videos/span_1080p.mp4",
        "benchmark_output": {
          "width": 3840,
          "height": 2160,
          "bitrate_mbps": 60
        }
      }
    ]
  }
}
""",
        encoding="utf-8",
    )

    plan, _ = build_plan(parsed)

    assert plan["comparison_class"] == "diagnostic"
    assert plan["parameters"]["frames"] == 120
    assert plan["parameters"]["gpu_video_trace"] is True
    assert plan["parameters"]["cpu_sampling"] is False
    assert plan["parameters"]["cuda_graph"] is False
    assert "Profiler overhead makes trace FPS non-publishable." in plan["limitations"]


def test_gpu_video_preflight_rejects_success_without_gpu() -> None:
    error = gpu_video_preflight_error(
        0,
        (
            "Could not find any NVIDIA GPUs. "
            "GPU video accelerator tracing is not available."
        ),
        gpu_id=0,
    )

    assert error is not None
    assert "GPU 0" in error


def test_gpu_memory_summaries_may_be_empty() -> None:
    assert {
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
    } == OPTIONAL_EMPTY_STATS_REPORTS
