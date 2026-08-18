from pathlib import Path

import pytest

from benchmarks.scripts.report.figures import load_published_data

TUNED_RESULTS = tuple(sorted(Path("benchmarks/results").glob("*/tuned.json")))


@pytest.mark.parametrize("tuned_path", TUNED_RESULTS, ids=lambda path: path.parent.name)
def test_loads_complete_published_matrix(tuned_path: Path) -> None:
    data = load_published_data(tuned_path.parent)

    assert len(data.revision) == 40
    assert int(data.revision, 16) > 0
    assert data.gpu.startswith("NVIDIA GeForce RTX ")
    assert data.cpu.startswith("AMD Ryzen ")
    assert data.power_limit_w > 0
    assert [(panel.workload, panel.variant) for panel in data.panels] == [
        ("RealESRGAN_x2plus", "720p"),
        ("RealESRGAN_x2plus", "1080p"),
        ("SPAN", "720p"),
        ("SPAN", "1080p"),
    ]
    assert all(
        {result.implementation for result in panel.results} == {"trtvideo", "vstrt", "vsgan"}
        for panel in data.panels
    )


@pytest.mark.parametrize("tuned_path", TUNED_RESULTS, ids=lambda path: path.parent.name)
def test_sweep_keeps_only_measured_eligible_points(tuned_path: Path) -> None:
    data = load_published_data(tuned_path.parent)
    real_720p, real_1080p, span_720p, _ = data.panels

    assert [point.streams for point in real_720p.sweep if point.implementation == "vstrt"] == [
        1,
        2,
        3,
        4,
        8,
    ]
    assert [point.streams for point in span_720p.sweep if point.implementation == "vsgan"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert [
        (limit.implementation, limit.streams, limit.kind) for limit in real_1080p.resource_limits
    ] == [
        ("vstrt", 8, "cuda-out-of-memory"),
        ("vsgan", 8, "cuda-out-of-memory"),
    ]
