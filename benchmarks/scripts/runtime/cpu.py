"""Process-attributed CPU accounting for benchmark child workloads."""

from __future__ import annotations

import os
import resource
from dataclasses import asdict, dataclass
from typing import Any


class CpuAccountingError(RuntimeError):
    """Raised when child CPU accounting produces an invalid measurement."""


@dataclass(frozen=True)
class ChildCpuSnapshot:
    """Cumulative CPU time of terminated and waited-for child processes."""

    user_time_sec: float
    system_time_sec: float


@dataclass(frozen=True)
class ChildCpuUsage:
    """CPU consumed by one measured subprocess tree."""

    user_time_sec: float
    system_time_sec: float
    total_time_sec: float
    average_cores: float
    available_logical_cpus: int
    capacity_percent: float
    accounting: str = "getrusage(RUSAGE_CHILDREN)"
    scope: str = "measured-child-process-tree"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_child_cpu() -> ChildCpuSnapshot:
    """Read cumulative CPU time for children already reaped by this process."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ChildCpuSnapshot(
        user_time_sec=float(usage.ru_utime),
        system_time_sec=float(usage.ru_stime),
    )


def available_logical_cpus() -> int:
    """Return CPUs available to the container process after affinity restrictions."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    if count <= 0:
        raise CpuAccountingError("No logical CPUs are available to the benchmark process")
    return count


def summarize_child_cpu(
    before: ChildCpuSnapshot,
    after: ChildCpuSnapshot,
    *,
    wall_time_sec: float,
    logical_cpus: int | None = None,
) -> ChildCpuUsage:
    """Convert cumulative child snapshots into one measured-run CPU report."""
    if wall_time_sec <= 0:
        raise CpuAccountingError("Measured wall time must be positive")
    cpu_count = logical_cpus if logical_cpus is not None else available_logical_cpus()
    if cpu_count <= 0:
        raise CpuAccountingError("Logical CPU count must be positive")

    user_time = after.user_time_sec - before.user_time_sec
    system_time = after.system_time_sec - before.system_time_sec
    if user_time < 0 or system_time < 0:
        raise CpuAccountingError("Child CPU counters moved backwards")
    total_time = user_time + system_time
    average_cores = total_time / wall_time_sec
    return ChildCpuUsage(
        user_time_sec=user_time,
        system_time_sec=system_time,
        total_time_sec=total_time,
        average_cores=average_cores,
        available_logical_cpus=cpu_count,
        capacity_percent=average_cores / cpu_count * 100,
    )
