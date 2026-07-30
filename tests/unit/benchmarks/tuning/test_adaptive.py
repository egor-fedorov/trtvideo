from __future__ import annotations

from benchmarks.scripts.tuning.adaptive import (
    CandidatePoint,
    has_confirmed_decline,
    select_peak_equivalent,
    sentinel_recovers,
    shortlist,
    upper_boundary_unresolved,
)
from benchmarks.scripts.tuning.contract import TunedCandidate


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
