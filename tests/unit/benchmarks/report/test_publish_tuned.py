import hashlib
from pathlib import Path

import pytest

from benchmarks.scripts.report.publish_tuned import (
    EvidenceSource,
    PublicationError,
    _tensor_set_digest,
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
