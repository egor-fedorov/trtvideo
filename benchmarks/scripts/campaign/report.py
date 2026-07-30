"""Render validated campaign summaries for human review."""

from __future__ import annotations

from typing import Any

from benchmarks.scripts.campaign.core import IMPLEMENTATIONS


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a validated campaign summary as Markdown."""
    lines = [
        "# Rotated Benchmark Campaign",
        "",
        f"Status: `{summary['status']}`. Publication ready: "
        f"`{'yes' if summary['publication']['ready'] else 'no'}`.",
        f"Execution profile: `{summary['execution_profile']}`.",
        "",
        "| Implementation | Runs | Median FPS | vs trtvideo | Median wall, s | "
        "CPU cores | CPU capacity, % | GPU util, % | Power, W | J/frame | "
        "Peak VRAM, MiB | Bitrate, Mbps | Size, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        lines.append(
            f"| {result['product']} | {summary['parameters']['rounds']} | "
            f"{stats['median_fps']:.3f} | "
            f"{result['relative_to_trtvideo_percent']:+.2f}% | "
            f"{stats['median_wall_time_sec']:.2f} | "
            f"{stats['median_cpu_cores']:.3f} | "
            f"{stats['median_cpu_capacity_percent']:.2f} | "
            f"{stats['median_gpu_utilization_percent']:.2f} | "
            f"{stats['median_power_w']:.2f} | "
            f"{stats['median_joules_per_frame']:.3f} | "
            f"{stats['median_peak_vram_mib']:.1f} | "
            f"{stats['median_output_bitrate_mbps']:.3f} | "
            f"{stats['median_output_size_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Implementation | Startup, s | Steady-state frame loop, s | Finalize + mux, s |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        lines.append(
            f"| {result['product']} | {stats['median_startup_sec']:.3f} | "
            f"{stats['median_steady_state_frame_loop_sec']:.3f} | "
            f"{stats['median_finalize_mux_sec']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Implementation | Stability | Full spread | 4-of-5 spread | Outlier | Raw FPS |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for name in IMPLEMENTATIONS:
        result = summary["implementations"][name]
        stats = result["statistics"]
        stability = result["stability"]
        consensus = stability["consensus"]
        outlier = stability["outlier"]
        consensus_spread = f"{consensus['relative_spread']:.2%}" if consensus else "-"
        outlier_label = f"round {outlier['round']}: {outlier['fps']:.3f} FPS" if outlier else "-"
        raw_values = ", ".join(f"{value:.3f}" for value in stats["values_fps"])
        lines.append(
            f"| {result['product']} | {stability['status']} | "
            f"{stability['full_relative_spread']:.2%} | {consensus_spread} | "
            f"{outlier_label} | {raw_values} |"
        )
    if summary["publication"]["warnings"]:
        lines.extend(["", "Publication warnings:"])
        lines.extend(f"- {warning}" for warning in summary["publication"]["warnings"])
    if summary["publication"]["errors"]:
        lines.extend(["", "Publication gaps:"])
        lines.extend(f"- {gap}" for gap in summary["publication"]["errors"])
    lines.append("")
    return "\n".join(lines)
