import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.scripts.report.publish_tuned import (
    CANONICAL_WORKLOADS,
    EvidenceSource,
    PublicationError,
    _identity_interpretation,
    _intra_session_reproducibility,
    _run_observations,
    _tensor_set_digest,
    _workloads_for_source,
)


def test_tensor_set_digest_is_order_independent() -> None:
    artifacts = [
        {"stage": "output", "frame_index": 1, "sha256": "b" * 64},
        {"stage": "input", "frame_index": 0, "sha256": "a" * 64},
    ]
    payload = f"input\t0\t{'a' * 64}\noutput\t1\t{'b' * 64}\n"
    expected = hashlib.sha256(payload.encode()).hexdigest()

    assert _tensor_set_digest({"artifacts": artifacts}) == expected
    assert _tensor_set_digest({"artifacts": list(reversed(artifacts))}) == expected


def test_evidence_source_maps_only_canonical_paths(tmp_path: Path) -> None:
    source = EvidenceSource(tmp_path)
    evidence = tmp_path / "workload" / "selection.json"

    assert source.canonical(evidence) == (
        "artefacts/benchmarks/comparative/tuning/workload/selection.json"
    )
    assert source.resolve(source.canonical(evidence)) == evidence

    with pytest.raises(PublicationError, match="escapes tuned evidence"):
        source.resolve("artefacts/benchmarks/project/suite.json")


def test_publication_requires_complete_canonical_media_contract(tmp_path: Path) -> None:
    for base in {item[0] for item in CANONICAL_WORKLOADS}:
        (tmp_path / f"{base}-matrix.json").write_text("{}\n", encoding="utf-8")

    assert _workloads_for_source(EvidenceSource(tmp_path)) == CANONICAL_WORKLOADS


def test_publication_rejects_incomplete_canonical_media_contract(tmp_path: Path) -> None:
    (tmp_path / "realesrgan_x2plus_madrid-matrix.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="complete canonical Madrid matrix"):
        _workloads_for_source(EvidenceSource(tmp_path))


def test_identity_interpretation_does_not_overclaim_mixed_outputs() -> None:
    identities = [
        {
            "candidate_tensor_sha256_sets_identical": True,
            "candidate_mp4_sha256_identical": True,
        },
        {
            "candidate_tensor_sha256_sets_identical": False,
            "candidate_mp4_sha256_identical": False,
        },
    ]

    interpretation = _identity_interpretation(identities)

    assert "reported per workload" in interpretation
    assert "quality gates pass" in interpretation


def test_run_observations_summarize_all_campaign_rounds(tmp_path: Path) -> None:
    source = EvidenceSource(tmp_path)
    manifest_paths = []
    for index, (temperature, capped, reasons) in enumerate(
        ((54, False, ["gpu_idle"]), (58, True, ["sw_power_cap"])),
        start=1,
    ):
        path = tmp_path / f"round-{index:02d}" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "measured": {
                        "metrics": {
                            "nvml": {
                                "temperature": {"peak_c": temperature},
                                "power": {"power_cap_observed": capped},
                                "throttle_reasons": reasons,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest_paths.append(source.canonical(path))

    campaign = {
        "rounds": [{"manifests": {"trtvideo": manifest_path}} for manifest_path in manifest_paths]
    }

    assert _run_observations(source, campaign, "trtvideo") == {
        "peak_temperature_c": 58,
        "power_cap_observed": True,
        "throttle_reasons": ["gpu_idle", "sw_power_cap"],
    }


def test_intra_session_reproducibility_compares_selected_external_profiles() -> None:
    selection = {
        "winners": {
            "vstrt": {"median_fps": 10.0},
            "vsgan": {"median_fps": 20.0},
        }
    }
    campaign = {
        "implementations": {
            "vstrt": {"statistics": {"median_fps": 10.04}},
            "vsgan": {"statistics": {"median_fps": 19.96}},
        }
    }

    result = _intra_session_reproducibility(selection, campaign)

    assert result["max_absolute_delta_percent"] == pytest.approx(0.4)
    assert [item["implementation"] for item in result["comparisons"]] == ["vstrt", "vsgan"]
