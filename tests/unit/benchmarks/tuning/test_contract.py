from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scripts.tuning.contract import (
    TuningContract,
    TuningContractError,
    load_tuning_contract,
)


@pytest.mark.parametrize(
    ("path", "reconnaissance_frames", "warmup_frames", "bitrate_validation"),
    [
        (Path("benchmarks/tuning/candidates.json"), 300, 30, False),
        (Path("benchmarks/tuning/span_candidates.json"), 1000, 100, True),
    ],
)
def test_repository_tuning_contract_declares_adaptive_search(
    path: Path,
    reconnaissance_frames: int,
    warmup_frames: int,
    bitrate_validation: bool,
) -> None:
    contract = load_tuning_contract(path)

    assert list(contract.search.stream_range) == list(range(1, 9))
    assert contract.search.sentinel_streams == 8
    assert contract.search.decline_margin == 0.01
    assert contract.search.decline_patience == 2
    assert contract.search.shortlist_size == 3
    assert contract.search.reconnaissance.measured_frames == reconnaissance_frames
    assert contract.search.reconnaissance.warmup_frames == warmup_frames
    assert contract.search.reconnaissance.initial_runs == 1
    assert contract.search.reconnaissance.bitrate_validation is bitrate_validation
    assert contract.search.confirmation.measured_frames == 1000
    assert contract.search.confirmation.initial_runs == 3
    assert contract.search.confirmation.extra_runs_on_spread == 2
    assert contract.search.confirmation.spread_threshold == 0.01
    assert contract.search.confirmation.max_relative_spread == 0.05
    assert contract.search.confirmation.bitrate_validation is True
    assert contract.selection.metric == "median_end_to_end_fps"
    assert contract.selection.equivalence_margin == 0.01
    assert contract.selection.tie_breaker == "lowest_num_streams_then_graph_off"
    assert contract.project_profile.as_dict() == {
        "backend": "nvcodec",
        "cuda_graph": False,
    }


def test_candidate_arguments_are_explicit_and_deterministic() -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))

    assert (
        contract.candidate("vstrt-s3-g0").runner_arguments()
        == "--requests auto --num-streams 3 --vs-threads auto --no-cuda-graph"
    )
    assert (
        contract.candidate("vsgan-s5-tauto-g1").runner_arguments()
        == "--requests auto --num-streams 5 --vs-threads auto --cuda-graph"
    )


def test_realesrgan_reconnaissance_arguments_disable_only_bitrate_acceptance() -> None:
    contract = load_tuning_contract(Path("benchmarks/tuning/candidates.json"))

    assert contract.search.reconnaissance.runner_arguments() == (
        "--frames 300 --warmup-frames 30 --runs 1 --extra-runs 0 "
        "--spread-threshold 0.01 --max-relative-spread 0.05 "
        "--idle-seconds 0 --skip-bitrate-validation"
    )


def test_contract_rejects_sentinel_below_maximum() -> None:
    value = json.loads(Path("benchmarks/tuning/candidates.json").read_text(encoding="utf-8"))
    value["search"]["sentinel_streams"] = 7

    with pytest.raises(
        TuningContractError,
        match="sentinel_streams must equal maximum_streams",
    ):
        TuningContract.from_dict(value)


def test_contract_rejects_confirmation_that_is_not_canonical_length() -> None:
    value = json.loads(Path("benchmarks/tuning/candidates.json").read_text(encoding="utf-8"))
    value["search"]["confirmation"]["measured_frames"] = 0

    with pytest.raises(
        TuningContractError,
        match="confirmation.measured_frames must be a positive integer",
    ):
        TuningContract.from_dict(value)


def test_contract_rejects_extension_threshold_above_hard_limit() -> None:
    value = json.loads(Path("benchmarks/tuning/candidates.json").read_text(encoding="utf-8"))
    value["search"]["confirmation"]["spread_threshold"] = 0.06

    with pytest.raises(
        TuningContractError,
        match="spread_threshold cannot exceed max_relative_spread",
    ):
        TuningContract.from_dict(value)
