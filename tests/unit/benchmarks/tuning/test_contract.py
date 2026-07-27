from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scripts.tuning.contract import (
    TuningContract,
    TuningContractError,
    load_tuning_contract,
)


def test_repository_tuning_contract_declares_required_vstrt_sweep() -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))

    assert {
        candidate.num_streams
        for candidate in contract.for_implementation("vstrt")
    } >= {2, 3, 4}
    assert contract.selection.metric == "median_end_to_end_fps"
    assert contract.selection.max_relative_spread == 0.05
    assert contract.project_profile.as_dict() == {
        "backend": "nvcodec",
        "cuda_graph": False,
    }


def test_span_tuning_contract_contains_measured_interior_peak_range() -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/span_candidates.json"))

    assert {
        candidate.num_streams
        for candidate in contract.for_implementation("vstrt")
    } == {2, 3, 4, 5, 6}


def test_candidate_arguments_are_explicit_and_deterministic() -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))

    assert (
        contract.candidate("vstrt-s3-g0").runner_arguments()
        == "--requests auto --num-streams 3 --vs-threads auto --no-cuda-graph"
    )


def test_contract_rejects_incomplete_vstrt_stream_sweep() -> None:
    value = json.loads(
        Path("benchmarks/tuning/candidates.json").read_text(encoding="utf-8")
    )
    value["candidates"] = [
        candidate
        for candidate in value["candidates"]
        if candidate["id"] != "vstrt-s3-g0"
    ]

    with pytest.raises(TuningContractError, match="2, 3, and 4"):
        TuningContract.from_dict(value)
