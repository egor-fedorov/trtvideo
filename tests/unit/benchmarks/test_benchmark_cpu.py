from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_media.benchmarking import cpu
from ai_media.benchmarking.cpu import (
    ChildCpuSnapshot,
    CpuAccountingError,
    available_logical_cpus,
    snapshot_child_cpu,
    summarize_child_cpu,
)


def test_snapshot_reads_child_process_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpu.resource,
        "getrusage",
        lambda scope: SimpleNamespace(ru_utime=12.5, ru_stime=2.25),
    )

    assert snapshot_child_cpu() == ChildCpuSnapshot(
        user_time_sec=12.5,
        system_time_sec=2.25,
    )


def test_summary_reports_cpu_seconds_and_average_cores() -> None:
    usage = summarize_child_cpu(
        ChildCpuSnapshot(user_time_sec=10.0, system_time_sec=2.0),
        ChildCpuSnapshot(user_time_sec=120.0, system_time_sec=27.0),
        wall_time_sec=100.0,
        logical_cpus=16,
    )

    assert usage.user_time_sec == 110.0
    assert usage.system_time_sec == 25.0
    assert usage.total_time_sec == 135.0
    assert usage.average_cores == 1.35
    assert usage.capacity_percent == pytest.approx(8.4375)
    assert usage.available_logical_cpus == 16
    assert usage.scope == "measured-child-process-tree"


def test_available_cpu_count_uses_process_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: {2, 3, 4, 5})

    assert available_logical_cpus() == 4


def test_summary_rejects_invalid_counter_delta() -> None:
    with pytest.raises(CpuAccountingError, match="moved backwards"):
        summarize_child_cpu(
            ChildCpuSnapshot(user_time_sec=2.0, system_time_sec=1.0),
            ChildCpuSnapshot(user_time_sec=1.0, system_time_sec=1.0),
            wall_time_sec=1.0,
            logical_cpus=1,
        )
