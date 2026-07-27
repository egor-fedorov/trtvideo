"""External NVML sampling for end-to-end benchmark processes."""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any


class NvmlError(RuntimeError):
    """Raised when required NVML metrics are unavailable."""


@dataclass(frozen=True)
class NvmlSample:
    """One privacy-safe sample relative to the measured process start."""

    relative_sec: float
    power_w: float | None
    memory_used_mib: float | None
    gpu_utilization_percent: int | None
    memory_utilization_percent: int | None
    temperature_c: int | None
    graphics_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_limit_w: float | None
    compute_process_count: int | None
    graphics_process_count: int | None
    throttle_reasons: tuple[str, ...]


_THROTTLE_CONSTANTS = {
    "gpu_idle": "nvmlClocksThrottleReasonGpuIdle",
    "applications_clocks_setting": "nvmlClocksThrottleReasonApplicationsClocksSetting",
    "sw_power_cap": "nvmlClocksThrottleReasonSwPowerCap",
    "hw_slowdown": "nvmlClocksThrottleReasonHwSlowdown",
    "sync_boost": "nvmlClocksThrottleReasonSyncBoost",
    "sw_thermal_slowdown": "nvmlClocksThrottleReasonSwThermalSlowdown",
    "hw_thermal_slowdown": "nvmlClocksThrottleReasonHwThermalSlowdown",
    "hw_power_brake_slowdown": "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
    "display_clock_setting": "nvmlClocksThrottleReasonDisplayClockSetting",
}
_INVALID_THROTTLE_REASONS = {
    "hw_slowdown",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
}


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _safe_call(function, *args, default=None):
    try:
        return function(*args)
    except Exception:
        return default


def decode_throttle_reasons(nvml: ModuleType, mask: int | None) -> tuple[str, ...]:
    """Translate an NVML bit mask without persisting opaque numeric values."""
    if mask is None:
        return ()
    reasons = []
    for name, constant_name in _THROTTLE_CONSTANTS.items():
        constant = getattr(nvml, constant_name, 0)
        if constant and mask & constant:
            reasons.append(name)
    return tuple(sorted(reasons))


def summarize_samples(
    samples: list[NvmlSample],
    *,
    wall_time_sec: float,
    frames: int,
    max_compute_processes: int = 1,
    max_graphics_processes: int = 0,
) -> dict[str, Any]:
    """Integrate power and summarize external GPU state for one measured process."""
    errors: list[str] = []
    if wall_time_sec <= 0:
        errors.append("wall_time_sec must be greater than zero")
    if frames <= 0:
        errors.append("frames must be greater than zero")
    if max_compute_processes < 0 or max_graphics_processes < 0:
        errors.append("GPU process limits cannot be negative")
    if not samples:
        errors.append("NVML sampler produced no samples")
        return {"valid": False, "errors": errors, "sample_count": 0}

    ordered = sorted(samples, key=lambda sample: sample.relative_sec)
    before_start = [sample for sample in ordered if sample.relative_sec <= 0]
    after_end = [sample for sample in ordered if sample.relative_sec >= wall_time_sec]
    bounded = [sample for sample in ordered if 0 < sample.relative_sec < wall_time_sec]
    if before_start:
        bounded.insert(0, replace(before_start[-1], relative_sec=0.0))
    elif ordered:
        bounded.insert(0, replace(ordered[0], relative_sec=0.0))
    if after_end:
        bounded.append(replace(after_end[0], relative_sec=wall_time_sec))
    elif ordered:
        bounded.append(replace(ordered[-1], relative_sec=wall_time_sec))
    if not bounded:
        bounded = [ordered[0]]

    power_samples = [sample for sample in bounded if sample.power_w is not None]
    memory_samples = [sample for sample in bounded if sample.memory_used_mib is not None]
    if len(power_samples) < 2:
        errors.append("NVML power sampling requires at least two samples")
    if not memory_samples:
        errors.append("NVML memory usage is unavailable")

    energy_j = 0.0
    for previous, current in zip(power_samples, power_samples[1:], strict=False):
        elapsed = current.relative_sec - previous.relative_sec
        if elapsed > 0 and previous.power_w is not None and current.power_w is not None:
            energy_j += elapsed * (previous.power_w + current.power_w) / 2

    baseline_memory = memory_samples[0].memory_used_mib if memory_samples else None
    peak_memory = (
        max(
            sample.memory_used_mib
            for sample in memory_samples
            if sample.memory_used_mib is not None
        )
        if memory_samples
        else None
    )
    peak_delta = (
        max(0.0, peak_memory - baseline_memory)
        if peak_memory is not None and baseline_memory is not None
        else None
    )
    throttle_reasons = sorted(
        {reason for sample in bounded for reason in sample.throttle_reasons}
    )
    invalid_throttle = sorted(set(throttle_reasons) & _INVALID_THROTTLE_REASONS)
    if invalid_throttle:
        errors.append(f"Invalid throttle reasons observed: {', '.join(invalid_throttle)}")

    baseline = bounded[0]
    if baseline.compute_process_count not in {None, 0}:
        errors.append("Another compute process was active before the measured process")
    if baseline.graphics_process_count not in {None, 0}:
        errors.append("A graphics process was active on the benchmark GPU")
    compute_counts = [
        sample.compute_process_count
        for sample in bounded
        if sample.compute_process_count is not None
    ]
    graphics_counts = [
        sample.graphics_process_count
        for sample in bounded
        if sample.graphics_process_count is not None
    ]
    observed_max_compute_processes = max(compute_counts, default=None)
    observed_max_graphics_processes = max(graphics_counts, default=None)
    if not compute_counts:
        errors.append("NVML compute process accounting is unavailable")
    if not graphics_counts:
        errors.append("NVML graphics process accounting is unavailable")
    if (
        observed_max_compute_processes is not None
        and observed_max_compute_processes > max_compute_processes
    ):
        errors.append(
            "Compute process count exceeded declared limit: "
            f"expected at most {max_compute_processes}, got {observed_max_compute_processes}"
        )
    if (
        observed_max_graphics_processes is not None
        and observed_max_graphics_processes > max_graphics_processes
    ):
        errors.append(
            "Graphics process count exceeded declared limit: "
            f"expected at most {max_graphics_processes}, got {observed_max_graphics_processes}"
        )
    power_limits = sorted(
        {sample.power_limit_w for sample in bounded if sample.power_limit_w is not None}
    )
    if not power_limits:
        errors.append("GPU power limit is unavailable")
    if len(power_limits) > 1:
        errors.append("GPU power limit changed during the measured process")

    powers = [sample.power_w for sample in power_samples if sample.power_w is not None]
    temperatures = [
        sample.temperature_c for sample in bounded if sample.temperature_c is not None
    ]
    gpu_utilizations = [
        sample.gpu_utilization_percent
        for sample in bounded
        if sample.gpu_utilization_percent is not None
    ]
    graphics_clocks = [
        sample.graphics_clock_mhz
        for sample in bounded
        if sample.graphics_clock_mhz is not None
    ]
    memory_clocks = [
        sample.memory_clock_mhz for sample in bounded if sample.memory_clock_mhz is not None
    ]
    if not temperatures:
        errors.append("NVML temperature sampling is unavailable")
    if not gpu_utilizations:
        errors.append("NVML GPU utilization sampling is unavailable")
    if not graphics_clocks or not memory_clocks:
        errors.append("NVML clock sampling is unavailable")
    return {
        "valid": not errors,
        "errors": errors,
        "sample_count": len(bounded),
        "sample_interval_observed_ms": (
            1000 * wall_time_sec / (len(bounded) - 1) if len(bounded) > 1 else None
        ),
        "power": {
            "average_w": energy_j / wall_time_sec if wall_time_sec > 0 else None,
            "peak_w": max(powers, default=None),
            "energy_j": energy_j,
            "joules_per_frame": energy_j / frames if frames > 0 else None,
            "power_cap_observed": "sw_power_cap" in throttle_reasons,
            "limit_w": power_limits[0] if len(power_limits) == 1 else None,
        },
        "memory": {
            "baseline_used_mib": baseline_memory,
            "peak_used_mib": peak_memory,
            "peak_delta_mib": peak_delta,
        },
        "utilization": {
            "average_gpu_percent": (
                sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else None
            ),
            "peak_gpu_percent": max(gpu_utilizations, default=None),
        },
        "temperature": {
            "start_c": temperatures[0] if temperatures else None,
            "peak_c": max(temperatures, default=None),
        },
        "clocks": {
            "graphics_min_mhz": min(graphics_clocks, default=None),
            "graphics_max_mhz": max(graphics_clocks, default=None),
            "memory_min_mhz": min(memory_clocks, default=None),
            "memory_max_mhz": max(memory_clocks, default=None),
        },
        "processes": {
            "max_compute_count": observed_max_compute_processes,
            "max_graphics_count": observed_max_graphics_processes,
            "compute_count_limit": max_compute_processes,
            "graphics_count_limit": max_graphics_processes,
        },
        "throttle_reasons": throttle_reasons,
    }


def write_samples(path: Path, samples: list[NvmlSample]) -> None:
    """Write raw samples as JSON Lines without process or device identifiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for sample in samples:
            value = asdict(sample)
            value["throttle_reasons"] = list(sample.throttle_reasons)
            json.dump(value, output, sort_keys=True)
            output.write("\n")


class NvmlSampler:
    """Background NVML sampler bound to one visible GPU index."""

    def __init__(self, gpu_id: int, interval_ms: int = 100):
        if interval_ms <= 0:
            raise ValueError("NVML sample interval must be greater than zero")
        self.gpu_id = gpu_id
        self.interval_sec = interval_ms / 1000
        self._nvml: ModuleType | None = None
        self._handle: Any = None
        self._samples: list[NvmlSample] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def initialize(self) -> dict[str, Any]:
        """Initialize NVML and return a sanitized static GPU snapshot."""
        try:
            nvml = importlib.import_module("pynvml")
        except ModuleNotFoundError as exc:
            raise NvmlError(
                "nvidia-ml-py is not installed; use the benchmark optional extra "
                "or the trtvideo:benchmark Docker image"
            ) from exc
        try:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as exc:
            raise NvmlError(f"Failed to initialize NVML for GPU {self.gpu_id}: {exc}") from exc
        self._nvml = nvml
        self._handle = handle
        memory = _safe_call(nvml.nvmlDeviceGetMemoryInfo, handle)
        power_limit = _safe_call(nvml.nvmlDeviceGetPowerManagementLimit, handle)
        compute_capability = _safe_call(
            getattr(nvml, "nvmlDeviceGetCudaComputeCapability", lambda *_: None),
            handle,
        )
        return {
            "index": self.gpu_id,
            "name": _decode(_safe_call(nvml.nvmlDeviceGetName, handle, default="unknown")),
            "compute_capability": (
                f"{compute_capability[0]}.{compute_capability[1]}"
                if compute_capability is not None
                else None
            ),
            "total_memory_mib": memory.total / (1024 * 1024) if memory is not None else None,
            "power_limit_w": (
                power_limit / 1000 if power_limit is not None else None
            ),
            "persistence_mode": _safe_call(
                getattr(nvml, "nvmlDeviceGetPersistenceMode", lambda *_: None),
                handle,
            ),
            "driver_version": _decode(
                _safe_call(nvml.nvmlSystemGetDriverVersion, default="unknown")
            ),
        }

    def _process_count(self, function_name: str) -> int | None:
        """Count unique process IDs rather than NVML process records."""
        if self._nvml is None:
            return None
        function = getattr(self._nvml, function_name, None)
        if function is None:
            return None
        processes = _safe_call(function, self._handle)
        return (
            len({int(process.pid) for process in processes})
            if processes is not None
            else None
        )

    def _sample(self) -> NvmlSample:
        if self._nvml is None or self._handle is None:
            raise NvmlError("NVML sampler is not initialized")
        nvml = self._nvml
        utilization = _safe_call(nvml.nvmlDeviceGetUtilizationRates, self._handle)
        memory = _safe_call(nvml.nvmlDeviceGetMemoryInfo, self._handle)
        throttle_mask = _safe_call(
            getattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", lambda *_: None),
            self._handle,
        )
        power_usage = _safe_call(nvml.nvmlDeviceGetPowerUsage, self._handle)
        power_limit = _safe_call(nvml.nvmlDeviceGetPowerManagementLimit, self._handle)
        return NvmlSample(
            relative_sec=max(0.0, time.perf_counter() - self._start_time),
            power_w=power_usage / 1000 if power_usage is not None else None,
            memory_used_mib=memory.used / (1024 * 1024) if memory is not None else None,
            gpu_utilization_percent=(utilization.gpu if utilization is not None else None),
            memory_utilization_percent=(utilization.memory if utilization is not None else None),
            temperature_c=_safe_call(
                nvml.nvmlDeviceGetTemperature,
                self._handle,
                getattr(nvml, "NVML_TEMPERATURE_GPU", 0),
            ),
            graphics_clock_mhz=_safe_call(
                nvml.nvmlDeviceGetClockInfo,
                self._handle,
                getattr(nvml, "NVML_CLOCK_GRAPHICS", 0),
            ),
            memory_clock_mhz=_safe_call(
                nvml.nvmlDeviceGetClockInfo,
                self._handle,
                getattr(nvml, "NVML_CLOCK_MEM", 2),
            ),
            power_limit_w=power_limit / 1000 if power_limit is not None else None,
            compute_process_count=self._process_count("nvmlDeviceGetComputeRunningProcesses"),
            graphics_process_count=self._process_count("nvmlDeviceGetGraphicsRunningProcesses"),
            throttle_reasons=decode_throttle_reasons(nvml, throttle_mask),
        )

    def sample_now(self) -> NvmlSample:
        """Collect and retain one sample."""
        sample = self._sample()
        with self._samples_lock:
            self._samples.append(sample)
        return sample

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            try:
                self.sample_now()
            except NvmlError:
                return

    def start(self, start_time: float) -> None:
        """Start sampling relative to an externally owned monotonic timer."""
        if self._nvml is None:
            raise NvmlError("NVML sampler must be initialized before start")
        self._start_time = start_time
        self._samples = []
        self._stop_event.clear()
        self.sample_now()
        self._thread = threading.Thread(target=self._sample_loop, name="nvml-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[NvmlSample]:
        """Stop the background sampler and include one final sample."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2))
            self._thread = None
        self.sample_now()
        with self._samples_lock:
            return list(self._samples)

    def samples_relative_to(self, samples: list[NvmlSample], start_time: float) -> list[NvmlSample]:
        """Rebase sampler-relative timestamps to the measured process timer."""
        offset = start_time - self._start_time
        return [
            replace(sample, relative_sec=sample.relative_sec - offset) for sample in samples
        ]

    def shutdown(self) -> None:
        """Release NVML state."""
        if self._nvml is not None:
            _safe_call(self._nvml.nvmlShutdown)
        self._nvml = None
        self._handle = None
