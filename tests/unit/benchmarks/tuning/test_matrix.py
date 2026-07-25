from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.tuning.matrix import TunedMatrixError, verify_matrix


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolution_evidence(
    root: Path,
    variant: str,
    *,
    revision: str = "a" * 40,
) -> Path:
    directory = root / "artefacts" / variant
    quality = {}
    for name in ("model_space", "product_output"):
        path = directory / f"{name}.json"
        _write_json(
            path,
            {
                "status": "valid",
                "publishable": True,
                "variant": variant,
            },
        )
        quality[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
    campaign_path = directory / "campaign.json"
    _write_json(
        campaign_path,
        {
            "status": "valid",
            "publishable": True,
            "comparison_profile": "tuned",
            "workload_id": "workload-v1",
            "benchmark_contract_version": 2,
            "variant": variant,
            "environment": {
                "repository_revision": revision,
                "gpu": {
                    "name": "NVIDIA GeForce RTX 3090",
                    "driver_version": "595.71",
                    "power_limit_w": 350.0,
                },
            },
        },
    )
    wrapper_path = directory / "final-campaign.json"
    _write_json(
        wrapper_path,
        {
            "document_type": "tuned-winner-campaign",
            "status": "valid",
            "publishable": False,
            "workload_id": "workload-v1",
            "variant": variant,
            "winners": {},
            "quality": quality,
            "campaign": {
                "path": campaign_path.relative_to(root).as_posix(),
                "sha256": _sha256(campaign_path),
            },
        },
    )
    return wrapper_path


def test_tuned_matrix_requires_both_valid_resolutions(tmp_path: Path) -> None:
    reports = {
        variant: _resolution_evidence(tmp_path, variant)
        for variant in ("720p", "1080p")
    }

    report = verify_matrix(root=tmp_path, campaign_reports=reports)

    assert report["status"] == "valid"
    assert report["publishable"] is True
    assert set(report["variants"]) == {"720p", "1080p"}


def test_tuned_matrix_rejects_missing_resolution(tmp_path: Path) -> None:
    with pytest.raises(TunedMatrixError, match="exactly 720p and 1080p"):
        verify_matrix(
            root=tmp_path,
            campaign_reports={
                "1080p": _resolution_evidence(tmp_path, "1080p")
            },
        )


def test_tuned_matrix_rejects_cross_resolution_revision_drift(
    tmp_path: Path,
) -> None:
    reports = {
        "720p": _resolution_evidence(tmp_path, "720p"),
        "1080p": _resolution_evidence(
            tmp_path,
            "1080p",
            revision="b" * 40,
        ),
    }

    with pytest.raises(TunedMatrixError, match="revision"):
        verify_matrix(root=tmp_path, campaign_reports=reports)
