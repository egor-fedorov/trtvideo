from __future__ import annotations

import pytest

from ai_media.benchmarking import nvml as nvml_module
from ai_media.benchmarking.nvml import NvmlError, NvmlSample, NvmlSampler, summarize_samples


def sample(
    relative_sec: float,
    power_w: float,
    *,
    memory_mib: float = 100,
    compute_processes: int = 1,
    throttle_reasons: tuple[str, ...] = (),
) -> NvmlSample:
    return NvmlSample(
        relative_sec=relative_sec,
        power_w=power_w,
        memory_used_mib=memory_mib,
        gpu_utilization_percent=80,
        memory_utilization_percent=20,
        temperature_c=60,
        graphics_clock_mhz=1500,
        memory_clock_mhz=7000,
        power_limit_w=250,
        compute_process_count=compute_processes,
        graphics_process_count=0,
        throttle_reasons=throttle_reasons,
    )


def test_summarize_samples_integrates_power_and_peak_memory() -> None:
    samples = [
        sample(0.0, 100, memory_mib=100, compute_processes=0),
        sample(0.5, 200, memory_mib=500),
        sample(1.0, 100, memory_mib=400),
    ]

    result = summarize_samples(samples, wall_time_sec=1.0, frames=100)

    assert result["valid"] is True
    assert result["power"]["energy_j"] == 150
    assert result["power"]["average_w"] == 150
    assert result["power"]["joules_per_frame"] == 1.5
    assert result["memory"]["peak_used_mib"] == 500
    assert result["memory"]["peak_delta_mib"] == 400


def test_summarize_samples_allows_fixed_power_cap() -> None:
    samples = [
        sample(0.0, 100, compute_processes=0),
        sample(1.0, 200, throttle_reasons=("sw_power_cap",)),
    ]

    result = summarize_samples(samples, wall_time_sec=1.0, frames=10)

    assert result["valid"] is True
    assert result["power"]["power_cap_observed"] is True


def test_summarize_samples_rejects_thermal_throttle() -> None:
    samples = [
        sample(0.0, 100, compute_processes=0),
        sample(1.0, 200, throttle_reasons=("sw_thermal_slowdown",)),
    ]

    result = summarize_samples(samples, wall_time_sec=1.0, frames=10)

    assert result["valid"] is False
    assert any("sw_thermal_slowdown" in error for error in result["errors"])


def test_summarize_samples_rejects_foreign_gpu_processes() -> None:
    samples = [
        sample(0.0, 100, compute_processes=1),
        sample(1.0, 200, compute_processes=2),
    ]

    result = summarize_samples(samples, wall_time_sec=1.0, frames=10)

    assert result["valid"] is False
    assert any("before" in error for error in result["errors"])
    assert any("declared limit" in error for error in result["errors"])


def test_summarize_samples_allows_declared_multiprocess_pipeline() -> None:
    samples = [
        sample(0.0, 100, compute_processes=0),
        sample(0.5, 200, compute_processes=2),
        sample(1.0, 180, compute_processes=2),
    ]

    result = summarize_samples(
        samples,
        wall_time_sec=1.0,
        frames=10,
        max_compute_processes=2,
    )

    assert result["valid"] is True
    assert result["processes"] == {
        "max_compute_count": 2,
        "max_graphics_count": 0,
        "compute_count_limit": 2,
        "graphics_count_limit": 0,
    }


def test_sampler_explains_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_module(_name: str):
        raise ModuleNotFoundError("No module named 'pynvml'")

    monkeypatch.setattr(nvml_module.importlib, "import_module", missing_module)

    with pytest.raises(NvmlError, match="benchmark optional extra"):
        NvmlSampler(gpu_id=0).initialize()
