from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.scripts.quality.product_output import (
    OutputEvidence,
    ProductOutputError,
    build_crop_command,
    build_metric_command,
    compare_product_outputs,
    crop_geometry,
    parse_metric_log,
    validate_evidence_set,
)
from benchmarks.scripts.runtime.environment import sha256_file


def _write_run_manifest(
    root: Path,
    *,
    name: str,
    product: str,
    engine_sha256: str,
    encoder: dict | None = None,
) -> Path:
    run_dir = root / f"artefacts/quality/{name}/run-01"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "output.mp4"
    output_path.write_bytes(f"{product} output".encode())
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "valid",
        "product": product,
        "workload_id": "workload-v1",
        "variant": "1080p",
        "parameters": {
            "frames": 1000,
            "encoder": encoder or {"codec": "h264", "rate_control": "cbr"},
        },
        "assets": {
            "input": {"sha256": "1" * 64},
            "onnx": {"sha256": "2" * 64},
            "engine": {"sha256": engine_sha256},
        },
        "reproducibility": {"publishable": True},
        "measured": {
            "validation": {"valid": True},
            "output": {
                "path": output_path.relative_to(root).as_posix(),
                "sha256": sha256_file(output_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _workload() -> dict:
    return {
        "id": "workload-v1",
        "clip": {"frames": 1000},
        "quality": {
            "product_output": {
                "frame_indices": [0, 499, 999],
                "thresholds": {
                    "psnr_min_db": 35.0,
                    "ssim_min": 0.95,
                },
                "crops": [
                    {
                        "name": "center",
                        "x": 0.375,
                        "y": 0.375,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ],
            }
        },
    }


def _variant() -> dict:
    return {
        "name": "1080p",
        "benchmark_output": {"width": 3840, "height": 2160},
    }


def test_parse_ffmpeg_metric_summaries() -> None:
    psnr = parse_metric_log(
        "psnr",
        "[Parsed_psnr_0] PSNR y:40 u:42 v:43 average:40.750 min:39 max:42",
    )
    ssim = parse_metric_log(
        "ssim",
        "[Parsed_ssim_0] SSIM Y:0.99 U:0.98 V:0.97 All:0.985 (18.2)",
    )

    assert psnr == {"exact": False, "average_db": 40.75}
    assert ssim == {"all": 0.985}
    assert parse_metric_log("psnr", "average:inf")["exact"] is True


def test_metric_command_decodes_both_complete_outputs(tmp_path: Path) -> None:
    command = build_metric_command(
        Path("/app/reference.mp4"),
        Path("/app/candidate.mp4"),
        metric="psnr",
        stats_path=tmp_path / "psnr.log",
    )

    assert command.count("-i") == 2
    assert "setpts=PTS-STARTPTS" in command[command.index("-filter_complex") + 1]
    assert command[-3:] == ["-f", "null", "-"]


def test_crop_geometry_and_command_use_one_decode(tmp_path: Path) -> None:
    geometry = crop_geometry(
        {"x": 0.375, "y": 0.375, "width": 0.25, "height": 0.25},
        output_width=3840,
        output_height=2160,
    )
    outputs = [
        (tmp_path / "first.png", 0, geometry),
        (tmp_path / "last.png", 999, geometry),
    ]

    command = build_crop_command(Path("/app/output.mp4"), outputs)

    assert geometry == (1440, 810, 960, 540)
    assert command.count("-i") == 1
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "split=2" in filter_graph
    assert "eq(n\\,999)" in filter_graph


def test_output_evidence_rejects_changed_file(tmp_path: Path) -> None:
    manifest_path = _write_run_manifest(
        tmp_path,
        name="trtvideo",
        product="trtvideo",
        engine_sha256="3" * 64,
    )
    output_path = tmp_path / "artefacts/quality/trtvideo/run-01/output.mp4"
    output_path.write_bytes(b"changed")

    with pytest.raises(ProductOutputError, match="SHA256 changed"):
        OutputEvidence.load(manifest_path, root=tmp_path)


def test_validate_evidence_requires_same_encoder(tmp_path: Path) -> None:
    reference = OutputEvidence.load(
        _write_run_manifest(
            tmp_path,
            name="trtvideo",
            product="trtvideo",
            engine_sha256="3" * 64,
        ),
        root=tmp_path,
    )
    vstrt = OutputEvidence.load(
        _write_run_manifest(
            tmp_path,
            name="vstrt",
            product="vs-mlrt",
            engine_sha256="3" * 64,
            encoder={"codec": "h264", "rate_control": "vbr"},
        ),
        root=tmp_path,
    )
    tas = OutputEvidence.load(
        _write_run_manifest(
            tmp_path,
            name="tas",
            product="TheAnimeScripter",
            engine_sha256="4" * 64,
        ),
        root=tmp_path,
    )

    with pytest.raises(ProductOutputError, match="changed encoder"):
        validate_evidence_set(reference, [vstrt, tas], expected_frames=1000)


def test_compare_product_outputs_builds_valid_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _write_run_manifest(
        tmp_path,
        name="trtvideo",
        product="trtvideo",
        engine_sha256="3" * 64,
    )
    candidates = [
        _write_run_manifest(
            tmp_path,
            name="vstrt",
            product="vs-mlrt",
            engine_sha256="3" * 64,
        ),
        _write_run_manifest(
            tmp_path,
            name="tas",
            product="TheAnimeScripter",
            engine_sha256="4" * 64,
        ),
    ]

    metric_directories = []
    crop_directories = {}

    def fake_metric(*args, metric: str, output_dir: Path, **kwargs) -> dict:
        metric_directories.append((output_dir, metric))
        if metric == "psnr":
            return {
                "exact": False,
                "average_db": 40.0,
                "frames": 1000,
                "stats_sha256": "5" * 64,
            }
        return {"all": 0.99, "frames": 1000, "stats_sha256": "6" * 64}

    monkeypatch.setattr(
        "benchmarks.scripts.quality.product_output.run_metric",
        fake_metric,
    )

    def fake_crops(evidence: OutputEvidence, *, output_dir: Path, **kwargs) -> list:
        crop_directories[evidence.product] = output_dir
        return []

    monkeypatch.setattr(
        "benchmarks.scripts.quality.product_output.generate_visual_crops", fake_crops
    )

    report = compare_product_outputs(
        reference,
        candidates,
        workload=_workload(),
        variant=_variant(),
        root=tmp_path,
        output_dir=tmp_path / "artefacts/quality/report",
    )

    assert report["status"] == "valid"
    assert report["publishable"] is True
    assert len(report["comparisons"]) == 2
    assert [item["implementation"] for item in report["comparisons"]] == [
        "vs-mlrt",
        "TheAnimeScripter",
    ]
    assert report["comparisons"][1]["engine_sha256"] == "4" * 64
    assert set(report["visual_crops"]) == {"trtvideo", "vs-mlrt", "TheAnimeScripter"}
    report_dir = tmp_path / "artefacts/quality/report"
    assert metric_directories == [
        (report_dir / candidate, metric)
        for candidate in ("vstrt", "tas")
        for metric in ("psnr", "ssim")
    ]
    assert crop_directories == {
        "trtvideo": report_dir / "crops/trtvideo",
        "vs-mlrt": report_dir / "crops/vstrt",
        "TheAnimeScripter": report_dir / "crops/tas",
    }


@pytest.mark.parametrize(
    "products",
    [
        [],
        ["vs-mlrt"],
        ["TheAnimeScripter"],
        ["vs-mlrt", "vs-mlrt"],
        ["vs-mlrt", "unknown"],
        ["vs-mlrt", "TheAnimeScripter", "TheAnimeScripter"],
    ],
)
def test_validate_evidence_requires_exact_candidate_set(
    tmp_path: Path, products: list[str]
) -> None:
    reference = OutputEvidence.load(
        _write_run_manifest(tmp_path, name="trtvideo", product="trtvideo", engine_sha256="3" * 64),
        root=tmp_path,
    )
    with pytest.raises(ProductOutputError, match="exactly vs-mlrt and TheAnimeScripter"):
        validate_evidence_set(
            reference,
            [replace(reference, product=product) for product in products],
            expected_frames=1000,
        )


def test_validate_evidence_requires_shared_vsmlrt_engine(tmp_path: Path) -> None:
    reference = OutputEvidence.load(
        _write_run_manifest(tmp_path, name="trtvideo", product="trtvideo", engine_sha256="3" * 64),
        root=tmp_path,
    )
    candidates = [
        replace(reference, product="vs-mlrt", engine_sha256="4" * 64),
        replace(reference, product="TheAnimeScripter", engine_sha256="5" * 64),
    ]
    with pytest.raises(ProductOutputError, match="vs-mlrt changed the shared engine"):
        validate_evidence_set(reference, candidates, expected_frames=1000)
