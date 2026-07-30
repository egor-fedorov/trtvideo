from pathlib import Path

from benchmarks.scripts.report.figures import load_published_data

RESULTS_DIR = Path("benchmarks/results/rtx-3090")


def test_loads_complete_published_matrix() -> None:
    data = load_published_data(RESULTS_DIR)

    assert len(data.revision) == 40
    assert int(data.revision, 16) > 0
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


def test_sweep_keeps_only_measured_eligible_points() -> None:
    data = load_published_data(RESULTS_DIR)
    real_720p, _, span_720p, _ = data.panels

    assert [point.streams for point in real_720p.sweep if point.implementation == "vstrt"] == [
        2,
        3,
        4,
    ]
    assert [point.streams for point in span_720p.sweep if point.implementation == "vsgan"] == [
        2,
        3,
        4,
        5,
        6,
    ]
