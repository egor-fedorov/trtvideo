import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.report.figures import (
    TAS_IO_ORDER,
    FigureDataError,
    _candidate_resources_from_json,
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
        {result.implementation for result in panel.results} == {"trtvideo", "vstrt", "tas"}
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
    assert [point.streams for point in span_720p.sweep if point.implementation == "vstrt"] == [
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
    ]
    assert all(
        [(point.decode_method, point.writer) for point in panel.tas_grid] == list(TAS_IO_ORDER)
        for panel in data.panels
    )
    if tuned_path.parent.name == "rtx-4090":
        assert real_720p.stream_winner_label == "Selected: 2 streams, graph on"
        assert data.panels[3].stream_winner_label == "Selected: 7 streams, graph on"


@pytest.mark.parametrize("tuned_path", TUNED_RESULTS, ids=lambda path: path.parent.name)
def test_published_figures_remain_byte_identical(tuned_path: Path) -> None:
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
        "median_cpu_cores": 2.0,
        "median_peak_vram_mib": 3072,
        "resource_measurement": {
            "stage": "reconnaissance",
            "frames_per_run": 300,
            "run_count": 1,
            "cpu_scope": "measured-child-process-tree",
            "cpu_accounting": "getrusage(RUSAGE_CHILDREN)",
            "vram_metric": "peak_delta_mib",
        },
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
        selection["candidates"] = [
            {
                **item,
                "median_cpu_cores": 3.0,
                "resource_measurement": {
                    **item["resource_measurement"],
                    "stage": "confirmation",
                    "frames_per_run": 1000,
                    "run_count": 3,
                },
            }
            for item in selection["reconnaissance"]
        ]
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


@pytest.mark.parametrize("status", ["invalid", "disqualified"])
def test_tas_preflight_exclusions_render_without_a_measurement(status: str) -> None:
    candidate = tas_candidate("cpu", "nelux", 0)
    points = _tas_grid_from_json(
        {
            "tas_preflight": {"candidates": [{**candidate, "status": status}]},
        }
    )
    assert len(points) == 1
    assert points[0].status == "quality failed"
    assert points[0].fps is None


def test_tas_figures_render_both_themes_and_actual_external_names(tas_results: Path) -> None:
    paths = generate_figures(tas_results, tas_results / "figures")
    assert len(paths) == 8
    for path in paths:
        svg = path.read_text()
        assert "VSGAN" not in svg
        assert "TAS" in svg and "vs-mlrt" in svg
        if path.name.startswith("tuned-sweep"):
            assert "TAS decoder / writer" in svg
            assert "vs-mlrt TensorRT streams" in svg
            assert "ffmpeg" in svg and "nelux" in svg
            assert "Selected:" in svg and "CUDA Graph off" in svg
    assert check_figures(tas_results, tas_results / "figures") == []


def test_legacy_vsgan_figures_keep_their_participants(tas_results: Path) -> None:
    path = tas_results / "tuned.json"
    document = json.loads(path.read_text())
    for workload in document["workloads"]:
        # This fixture is synthetic, not a relabeled published measurement.
        workload["final_campaign"]["results"][-1]["implementation"] = "vsgan"
        selection = workload["selection"]
        selection.pop("tas_preflight", None)
        selection["candidates"] = []
        streams = [
            item for item in selection["reconnaissance"] if item["implementation"] == "vstrt"
        ]
        selection["reconnaissance"] = streams + [
            {**item, "implementation": "vsgan", "candidate_id": item["candidate_id"] + "-legacy"}
            for item in streams
        ]
        for candidate in selection["reconnaissance"]:
            candidate.pop("resource_measurement", None)
    path.write_text(json.dumps(document))
    for figure in generate_figures(tas_results, tas_results / "figures"):
        svg = figure.read_text()
        assert "VSGAN" in svg and "TAS" not in svg
        assert "decoder / writer" not in svg


def test_resource_points_keep_stages_and_graph_profiles_separate(tas_results: Path) -> None:
    points = load_published_data(tas_results).panels[0].candidate_resources
    tas = [point for point in points if point.candidate_id == "tas-nvdec-nelux"]
    assert [(p.stage, p.frames, p.runs, p.cpu_cores) for p in tas] == [
        ("reconnaissance", 300, 1, 2.0),
        ("confirmation", 1000, 3, 3.0),
    ]
    assert all(point.winner for point in tas)
    assert all(point.profile["cuda_graph"] is True for point in tas)
    for path in generate_figures(tas_results, tas_results / "figures"):
        if path.name.startswith("tuning-resources"):
            svg = path.read_text()
            assert "above baseline (GiB)" in svg
            assert "vs-mlrt graph on" in svg and "TAS graph on" in svg
            stage = "Confirmation" if "confirmation" in path.name else "Reconnaissance"
            assert f"Tuning resources | {stage}" in svg


@pytest.mark.parametrize(
    "change", ["stage", "vram_metric", "missing_cpu", "nan", "negative", "duplicate"]
)
def test_resource_points_reject_mixed_or_missing_metrics(change: str) -> None:
    point = tas_candidate("cpu", "ffmpeg", 10)
    if change == "stage":
        point["resource_measurement"]["stage"] = "confirmation"
    elif change == "vram_metric":
        point["resource_measurement"]["vram_metric"] = "peak_used_mib"
    elif change == "missing_cpu":
        del point["median_cpu_cores"]
    elif change == "nan":
        point["median_peak_vram_mib"] = float("nan")
    elif change == "negative":
        point["median_cpu_cores"] = -1
    with pytest.raises(FigureDataError):
        _candidate_resources_from_json(
            {"reconnaissance": [point] * (2 if change == "duplicate" else 1)}
        )


def test_disqualified_candidate_has_no_resource_point() -> None:
    point = {**tas_candidate("cpu", "ffmpeg", 100), "status": "disqualified"}
    result = _candidate_resources_from_json({"reconnaissance": [point]})[0]
    assert result.cpu_cores is None and result.peak_vram_mib is None


def test_figure_data_rejects_duplicate_campaign_participants(tas_results: Path) -> None:
    path = tas_results / "tuned.json"
    document = json.loads(path.read_text())
    results = document["workloads"][0]["final_campaign"]["results"]
    results.append(results[-1])
    path.write_text(json.dumps(document))
    with pytest.raises(FigureDataError, match="Unexpected implementations"):
        load_published_data(tas_results)
