"""Verify that tuned publication evidence covers both canonical resolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_VARIANTS = ("720p", "1080p")


class TunedMatrixError(RuntimeError):
    """Raised when tuned resolution evidence cannot be published together."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TunedMatrixError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TunedMatrixError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TunedMatrixError(f"{label} has no artifact path")
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise TunedMatrixError(f"{label} escapes the repository root")
    return path


def _verified_artifact(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise TunedMatrixError(f"{label} evidence is missing")
    path = _repo_path(root, value.get("path"), label=label)
    expected_sha = value.get("sha256")
    if not path.is_file():
        raise TunedMatrixError(f"{label} artifact is missing: {path}")
    if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
        raise TunedMatrixError(f"{label} SHA256 changed")
    return path, _load_json(path)


def verify_matrix(
    *,
    root: Path,
    campaign_reports: dict[str, Path],
) -> dict[str, Any]:
    """Validate two final tuned resolution reports as one publication unit."""
    root = root.resolve()
    if set(campaign_reports) != set(REQUIRED_VARIANTS):
        raise TunedMatrixError("Tuned matrix requires exactly 720p and 1080p")

    entries: dict[str, dict[str, Any]] = {}
    shared_contract: tuple[Any, ...] | None = None
    for variant in REQUIRED_VARIANTS:
        wrapper_path = campaign_reports[variant].resolve()
        wrapper = _load_json(wrapper_path)
        checks = {
            "document type": (
                wrapper.get("document_type"),
                "tuned-winner-campaign",
            ),
            "status": (wrapper.get("status"), "valid"),
            "variant": (wrapper.get("variant"), variant),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise TunedMatrixError(
                    f"{variant} final campaign {label} changed "
                    f"({actual!r} != {expected!r})"
                )
        campaign_path, campaign = _verified_artifact(
            root,
            wrapper.get("campaign"),
            label=f"{variant} campaign",
        )
        if (
            campaign.get("status") != "valid"
            or campaign.get("publishable") is not True
            or campaign.get("comparison_profile") != "tuned"
        ):
            raise TunedMatrixError(f"{variant} tuned campaign is not publishable")
        if campaign.get("variant") != variant:
            raise TunedMatrixError(f"{variant} campaign variant changed")
        quality = wrapper.get("quality")
        if not isinstance(quality, dict):
            raise TunedMatrixError(f"{variant} full quality evidence is missing")
        quality_records = {}
        for name in ("model_space", "product_output"):
            quality_path, quality_report = _verified_artifact(
                root,
                quality.get(name),
                label=f"{variant} {name}",
            )
            if (
                quality_report.get("status") != "valid"
                or quality_report.get("publishable") is not True
                or quality_report.get("variant") != variant
            ):
                raise TunedMatrixError(
                    f"{variant} {name} quality evidence is not valid"
                )
            quality_records[name] = {
                "path": quality_path.relative_to(root).as_posix(),
                "sha256": _sha256(quality_path),
            }

        environment = campaign.get("environment", {})
        gpu = environment.get("gpu", {})
        contract = (
            campaign.get("workload_id"),
            campaign.get("benchmark_contract_version"),
            environment.get("repository_revision"),
            gpu.get("name"),
            gpu.get("driver_version"),
            gpu.get("power_limit_w"),
        )
        if shared_contract is None:
            shared_contract = contract
        elif contract != shared_contract:
            raise TunedMatrixError(
                "720p and 1080p evidence changed workload, revision, or GPU contract"
            )
        entries[variant] = {
            "winners": wrapper.get("winners"),
            "campaign": {
                "path": campaign_path.relative_to(root).as_posix(),
                "sha256": _sha256(campaign_path),
            },
            "quality": quality_records,
        }

    assert shared_contract is not None
    return {
        "schema_version": 1,
        "document_type": "tuned-publication-matrix",
        "status": "valid",
        "publishable": True,
        "scope": "cross-resolution-tuned-publication",
        "workload_id": shared_contract[0],
        "benchmark_contract_version": shared_contract[1],
        "environment": {
            "repository_revision": shared_contract[2],
            "gpu": {
                "name": shared_contract[3],
                "driver_version": shared_contract[4],
                "power_limit_w": shared_contract[5],
            },
        },
        "variants": entries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--campaign-720p", required=True)
    parser.add_argument("--campaign-1080p", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = verify_matrix(
            root=Path(args.root),
            campaign_reports={
                "720p": Path(args.campaign_720p),
                "1080p": Path(args.campaign_1080p),
            },
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TunedMatrixError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Tuned publication matrix valid: {output}")


if __name__ == "__main__":
    main()
