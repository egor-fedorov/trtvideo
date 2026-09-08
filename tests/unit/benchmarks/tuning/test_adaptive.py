from __future__ import annotations

import pytest

from benchmarks.scripts.tuning.adaptive import (
    CandidatePoint,
    has_confirmed_decline,
    resource_medians,
    select_peak_equivalent,
    sentinel_recovers,
    shortlist,
    upper_boundary_unresolved,
)
from benchmarks.scripts.tuning.contract import TasCandidate, TunedCandidate


def _point(streams: int, fps: float, *, graph: bool = False) -> CandidatePoint:
    candidate = TunedCandidate(
        candidate_id=f"vstrt-s{streams}-g{int(graph)}",
        implementation="vstrt",
        requests="auto",
        num_streams=streams,
        vapoursynth_threads="auto",
        cuda_graph=graph,
    )
    return CandidatePoint(
        candidate=candidate,
        median_fps=fps,
        relative_spread=0.002,
        suite_path=f"s{streams}-g{int(graph)}/suite.json",
    )


def test_decline_requires_two_materially_slower_trailing_points() -> None:
    points = [_point(1, 10), _point(2, 12), _point(3, 11.7), _point(4, 11.5)]

    assert has_confirmed_decline(points, relative_margin=0.01, patience=2)
    assert not has_confirmed_decline(points[:3], relative_margin=0.01, patience=2)


def test_sentinel_recovery_requires_material_improvement() -> None:
    points = [_point(1, 10), _point(2, 12), _point(3, 11), _point(4, 10.8)]

    assert sentinel_recovers(points, _point(8, 12.2), relative_margin=0.01)
    assert not sentinel_recovers(points, _point(8, 12.1), relative_margin=0.01)


def test_shortlist_uses_reconnaissance_only_as_a_coarse_ranking() -> None:
    points = [_point(1, 9), _point(2, 12), _point(3, 11.8), _point(4, 11.9)]

    assert [candidate.num_streams for candidate in shortlist(points, size=3)] == [
        2,
        4,
        3,
    ]


def test_materially_increasing_upper_boundary_requires_a_larger_range() -> None:
    assert upper_boundary_unresolved(
        [_point(1, 10), _point(2, 11), _point(3, 12)],
        relative_margin=0.01,
    )
    assert not upper_boundary_unresolved(
        [_point(1, 10), _point(2, 12), _point(3, 12.1)],
        relative_margin=0.01,
    )


def test_peak_equivalence_favors_competitor_resource_efficiency() -> None:
    points = [_point(5, 25.0), _point(6, 25.2), _point(6, 25.25, graph=True)]

    selected = select_peak_equivalent(points, equivalence_margin=0.01)

    assert selected is not None
    assert selected.candidate.num_streams == 5
    assert selected.candidate.cuda_graph is False


def _tas_point(decoder, writer, fps, vram, cpu) -> CandidatePoint:
    return CandidatePoint(
        candidate=TasCandidate(f"tas-{decoder}-{writer}", decoder, writer),
        median_fps=fps,
        relative_spread=0,
        suite_path="suite.json",
        median_peak_vram_mib=vram,
        median_cpu_cores=cpu,
    )


def test_tas_shortlist_uses_fps_not_resources() -> None:
    points = [
        _tas_point("cpu", "ffmpeg", 50, 1, 1),
        _tas_point("cpu", "nelux", 70, 100, 4),
        _tas_point("nvdec", "ffmpeg", 60, 200, 8),
        _tas_point("nvdec", "nelux", 90, 300, 9),
    ]
    assert [candidate.candidate_id for candidate in shortlist(points, size=3)] == [
        "tas-nvdec-nelux",
        "tas-cpu-nelux",
        "tas-nvdec-ffmpeg",
    ]


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (
            [
                _tas_point("nvdec", "nelux", 100, 500, 1),
                _tas_point("cpu", "ffmpeg", 99, 400, 8),
                _tas_point("cpu", "nelux", 98.99, 1, 1),
            ],
            "tas-cpu-ffmpeg",
        ),
        (
            [
                _tas_point("cpu", "ffmpeg", 100, 500, 8),
                _tas_point("nvdec", "nelux", 99.5, 500, 1),
            ],
            "tas-nvdec-nelux",
        ),
        (
            [
                _tas_point("nvdec", "nelux", 100, 500, 1),
                _tas_point("cpu", "ffmpeg", 99.5, 500, 1),
            ],
            "tas-cpu-ffmpeg",
        ),
    ],
)
def test_tas_peak_tie_break_is_vram_then_cpu_then_stable_id(points, expected) -> None:
    selected = select_peak_equivalent(points, equivalence_margin=0.01)
    assert selected is not None
    assert selected.candidate.candidate_id == expected


@pytest.mark.parametrize("vram", [None, float("nan"), float("inf"), -1])
def test_tas_tie_break_rejects_missing_or_invalid_resources(vram) -> None:
    with pytest.raises(ValueError, match="TAS resource"):
        select_peak_equivalent([_tas_point("cpu", "ffmpeg", 100, vram, 1)], equivalence_margin=0.01)


def test_tas_resource_medians_come_from_run_measurements() -> None:
    manifests = [
        {
            "measured": {
                "metrics": {
                    "nvml": {"memory": {"peak_delta_mib": vram}},
                    "cpu": {"average_cores": cpu},
                }
            }
        }
        for vram, cpu in [(1000, 2), (900, 4), (1200, 3)]
    ]
    assert resource_medians(manifests) == (1000.0, 3.0)
