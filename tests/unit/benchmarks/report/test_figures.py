import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.report.figures import (
    TAS_IO_ORDER,
    FigureDataError,
    _tas_grid_from_json,
    check_figures,
    generate_figures,
    load_published_data,
)

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


@pytest.mark.parametrize("tuned_path", TUNED_RESULTS, ids=lambda path: path.parent.name)
def test_historical_figures_remain_byte_identical(tuned_path: Path) -> None:
    assert check_figures(tuned_path.parent, tuned_path.parent / "figures") == []


def tas_candidate(decode: str, writer: str, fps: float) -> dict[str, Any]:
    return {
        "candidate_id": f"tas-{decode}-{writer}",
        "implementation": "tas",
        "execution_profile": {
            "execution_profile": "tuned",
            "decode_method": decode,
            "writer": writer,
            "cuda_graph": True,
        },
        "status": "eligible",
        "median_fps": fps,
    }


@pytest.fixture
def tas_results(tmp_path: Path) -> Path:
    # Synthetic TAS data exercises the new layout without relabeling a real measurement.
    document = json.loads(TUNED_RESULTS[0].read_text())
    for index, workload in enumerate(document["workloads"]):
        campaign = workload["final_campaign"]
        campaign["results"] = [
            {"implementation": name, "fps_median": fps, "cpu_cores": cpu, "peak_vram_mib": vram}
            for name, fps, cpu, vram in (
                ("trtvideo", 10, 1, 1024),
                ("vstrt", 9 if index == 0 else 12, 4, 2048),
                ("tas", 11, 2, 3072),
            )
        ]
        selection = workload["selection"]
        selection["reconnaissance"] = [
            item for item in selection["reconnaissance"] if item["implementation"] == "vstrt"
        ] + [tas_candidate(decode, writer, 8 + index) for decode, writer in TAS_IO_ORDER]
        selection["winners"] = {
            "vstrt": selection["winners"]["vstrt"],
            "tas": {"candidate_id": "tas-nvdec-nelux"},
        }
        selection["search"]["resource_limits"] = {"vstrt": None, "tas": []}
    (tmp_path / "tuned.json").write_text(json.dumps(document))
    return tmp_path


def test_tas_grid_has_categories_not_streams(tas_results: Path) -> None:
    data = load_published_data(tas_results)
    panel = data.panels[0]
    assert {point.implementation for point in panel.sweep} == {"vstrt"}
    assert [(point.decode_method, point.writer) for point in panel.tas_grid] == list(TAS_IO_ORDER)
    assert [point.candidate_id for point in panel.tas_grid if point.winner] == ["tas-nvdec-nelux"]
    assert panel.fastest_external().implementation == "tas"
    assert data.panels[1].fastest_external().implementation == "vstrt"


def test_tas_oom_and_disqualified_points_have_no_fps() -> None:
    rejected = {**tas_candidate("cpu", "ffmpeg", 500), "status": "disqualified"}
    limit = {
        "candidate_id": "tas-nvdec-nelux",
        "execution_profile": tas_candidate("nvdec", "nelux", 0)["execution_profile"],
        "kind": "cuda-out-of-memory",
    }
    points = _tas_grid_from_json(
        {
            "reconnaissance": [rejected],
            "search": {"resource_limits": {"tas": [limit]}},
        }
    )
    assert [(point.status, point.fps) for point in points] == [
        ("disqualified", None),
        ("OOM", None),
    ]


def test_tas_grid_rejects_duplicate_profiles() -> None:
    candidate = tas_candidate("cpu", "ffmpeg", 8)
    with pytest.raises(FigureDataError, match="duplicate TAS"):
        _tas_grid_from_json({"reconnaissance": [candidate, candidate]})


def test_tas_preflight_exclusions_render_without_a_measurement() -> None:
    candidate = tas_candidate("cpu", "nelux", 0)
    points = _tas_grid_from_json(
        {
            "tas_preflight": {"candidates": [{**candidate, "status": "invalid"}]},
        }
    )
    assert len(points) == 1
    assert points[0].status == "quality failed"
    assert points[0].fps is None


def test_tas_figures_render_both_themes_and_actual_external_names(tas_results: Path) -> None:
    paths = generate_figures(tas_results, tas_results / "figures")
    assert len(paths) == 4
    for path in paths:
        svg = path.read_text()
        assert "VSGAN" not in svg
        assert "TAS" in svg and "vs-mlrt" in svg
        if path.name.startswith("tuned-sweep"):
            assert "TAS decoder / writer" in svg
            assert "vs-mlrt TensorRT streams" in svg
            assert "ffmpeg" in svg and "nelux" in svg
    assert check_figures(tas_results, tas_results / "figures") == []


def test_figure_data_rejects_duplicate_campaign_participants(tas_results: Path) -> None:
    path = tas_results / "tuned.json"
    document = json.loads(path.read_text())
    results = document["workloads"][0]["final_campaign"]["results"]
    results.append(results[-1])
    path.write_text(json.dumps(document))
    with pytest.raises(FigureDataError, match="Unexpected implementations"):
        load_published_data(tas_results)
