from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.scripts.quality.capture_vspipe import (
    build_capture_command,
    normalize_vapoursynth_rgbs,
    serialize_vapoursynth_rgbs,
)
from benchmarks.scripts.quality.model_space import (
    CAPTURE_STAGE_CONTRACTS,
    CaptureManifest,
    ModelSpaceError,
    TensorThresholds,
    compare_inference_captures,
    compare_preprocessing_captures,
    create_tensor_artifact,
    evaluate_metrics,
    parse_frame_indices,
    write_capture_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_capture(
    root: Path,
    *,
    implementation: str,
    capture_scope: str,
    engine_sha256: str,
    input_value: float,
    output_value: float = 0.5,
    execution_profile: dict[str, object] | None = None,
    canonical_reference: Path | None = None,
) -> Path:
    root.mkdir(parents=True)
    artifacts = []
    values = {"input": input_value, "output": output_value}
    for stage in CAPTURE_STAGE_CONTRACTS[capture_scope]:
        path = root / f"{stage}.frame-000000.f32"
        np.full((3, 2, 2), values[stage], dtype="<f4").tofile(path)
        artifacts.append(
            create_tensor_artifact(
                stage=stage,
                frame_index=0,
                shape=(3, 2, 2),
                path=path,
                root=root,
            )
        )
    manifest_path = root / "manifest.json"
    write_capture_manifest(
        manifest_path,
        implementation=implementation,
        capture_scope=capture_scope,
        workload_id="workload-v1",
        variant="720p",
        input_sha256="1" * 64,
        onnx_sha256="2" * 64,
        engine_sha256=engine_sha256,
        image={
            "reference": f"{implementation}:test",
            "id": f"{implementation}-image",
            "repository_revision": "revision-1",
            "source_dirty": "0",
        },
        execution_profile=execution_profile
        or {
            "execution_profile": "upstream-default",
            "cuda_graph": False,
        },
        artifacts=artifacts,
        canonical_input_manifest_sha256=(
            _sha256(canonical_reference) if canonical_reference is not None else None
        ),
    )
    return manifest_path


def _output_thresholds(limit: float = 0.01) -> TensorThresholds:
    return TensorThresholds(
        p99_abs=limit,
        rmse=limit,
        min_psnr_db=40.0,
    )


def test_parse_frame_indices_sorts_and_deduplicates() -> None:
    assert parse_frame_indices("999,0,499,499", frame_count=1000) == (0, 499, 999)


def test_parse_frame_indices_rejects_out_of_range() -> None:
    with pytest.raises(ModelSpaceError, match="must stay"):
        parse_frame_indices("0,1000", frame_count=1000)


def test_capture_manifest_detects_changed_tensor(tmp_path: Path) -> None:
    manifest_path = _write_capture(
        tmp_path / "capture",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.25,
    )
    tensor_path = manifest_path.parent / "input.frame-000000.f32"
    tensor_path.write_bytes(b"\0" * tensor_path.stat().st_size)

    with pytest.raises(ModelSpaceError, match="SHA256 changed"):
        CaptureManifest.load(manifest_path)


def test_inference_parity_requires_exact_shared_input(tmp_path: Path) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        capture_scope="shared-input-inference",
        engine_sha256="3" * 64,
        input_value=0.2501,
        output_value=0.5001,
        canonical_reference=reference,
    )

    report = compare_inference_captures(
        reference,
        [candidate],
        output_thresholds=_output_thresholds(),
    )

    assert report["status"] == "invalid"
    assert report["acceptance_gate"] is True
    assert any("canonical input tensor differs" in error for error in report["errors"])


def test_inference_parity_accepts_close_output_from_exact_input(tmp_path: Path) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        capture_scope="shared-input-inference",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5001,
        canonical_reference=reference,
    )

    report = compare_inference_captures(
        reference,
        [candidate],
        output_thresholds=_output_thresholds(),
    )

    assert report["status"] == "valid"
    assert report["publishable"] is True
    assert report["comparisons"][0]["status"] == "valid"
    assert report["comparisons"][0]["tensors"][0]["stage"] == "input"
    assert report["comparisons"][0]["tensors"][0]["metrics"]["exact"] is True


def test_inference_parity_requires_byte_exact_not_only_numeric_input(
    tmp_path: Path,
) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.0,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        capture_scope="shared-input-inference",
        engine_sha256="3" * 64,
        input_value=-0.0,
        canonical_reference=reference,
    )

    report = compare_inference_captures(
        reference,
        [candidate],
        output_thresholds=_output_thresholds(),
    )

    input_metrics = report["comparisons"][0]["tensors"][0]["metrics"]
    assert input_metrics["exact"] is True
    assert input_metrics["sha256_equal"] is False
    assert report["status"] == "invalid"


def test_inference_parity_rejects_large_output_difference(tmp_path: Path) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="VSGAN-tensorrt-docker",
        capture_scope="shared-input-inference",
        engine_sha256="4" * 64,
        input_value=0.25,
        output_value=0.75,
        canonical_reference=reference,
    )

    report = compare_inference_captures(
        reference,
        [candidate],
        output_thresholds=_output_thresholds(),
    )

    assert report["status"] == "invalid"
    assert any("output frame 0" in error for error in report["errors"])


def test_preprocessing_difference_is_diagnostic_not_an_acceptance_failure(
    tmp_path: Path,
) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        capture_scope="production-reference",
        engine_sha256="3" * 64,
        input_value=0.25,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        capture_scope="production-preprocessing",
        engine_sha256="3" * 64,
        input_value=1.25,
    )

    report = compare_preprocessing_captures(reference, [candidate])

    assert report["status"] == "complete"
    assert report["publishable"] is True
    assert report["acceptance_gate"] is False
    comparison = report["comparisons"][0]
    assert comparison["status"] == "complete"
    assert comparison["tensors"][0]["metrics"]["rmse"] == pytest.approx(1.0)
    assert comparison["tensors"][0]["candidate_statistics"]["outside_unit_interval_fraction"] == 1.0


def test_maximum_error_is_diagnostic_not_an_acceptance_limit() -> None:
    errors = evaluate_metrics(
        {
            "finite": True,
            "exact": False,
            "max_abs": 0.25,
            "p99_abs": 0.005,
            "rmse": 0.002,
            "psnr_db": 50.0,
        },
        TensorThresholds(
            p99_abs=0.01,
            rmse=0.004,
            min_psnr_db=48.0,
        ),
    )

    assert errors == []


def _vspipe_args(*, implementation: str, execution_profile: str) -> argparse.Namespace:
    return argparse.Namespace(
        implementation=implementation,
        engine="/app/models/model.engine",
        gpu_id=0,
        execution_profile=execution_profile,
        requests="auto" if execution_profile == "tuned" else None,
        num_streams=3 if execution_profile == "tuned" else None,
        vs_threads=6 if execution_profile == "tuned" else None,
        cuda_graph=execution_profile == "tuned",
        script=f"/app/benchmarks/{implementation}/upscale.vpy",
    )


def test_vspipe_production_capture_uses_requested_video_frame(tmp_path: Path) -> None:
    command = build_capture_command(
        _vspipe_args(implementation="vsgan", execution_profile="upstream-default"),
        input_path=Path("/app/videos/input.mp4"),
        output_path=tmp_path / "output.f32",
        frame_index=499,
        stage="input",
    )

    assert command[command.index("--start") + 1] == "499"
    assert "source=/app/videos/input.mp4" in command
    assert "model_space_stage=input" in command
    assert "vs_threads=4" in command


def test_vspipe_shared_input_capture_bypasses_video_preprocessing(tmp_path: Path) -> None:
    command = build_capture_command(
        _vspipe_args(implementation="vstrt", execution_profile="tuned"),
        input_path=Path("/app/artefacts/input.nut"),
        output_path=tmp_path / "output.f32",
        frame_index=999,
        stage="output",
        shared_input=True,
    )

    assert command[command.index("--start") + 1] == "0"
    assert command[command.index("--end") + 1] == "0"
    assert "model_input=/app/artefacts/input.nut" in command
    assert not any(value.startswith("source=") for value in command)
    assert "num_streams=3" in command
    assert "cuda_graph=1" in command


def test_vapoursynth_plane_serialization_round_trips_rgb(tmp_path: Path) -> None:
    source = tmp_path / "source.f32"
    physical_gbr = tmp_path / "physical.f32"
    output = tmp_path / "output.f32"
    logical_rgb = np.concatenate(
        [
            np.full(4, 0.1, dtype="<f4"),
            np.full(4, 0.2, dtype="<f4"),
            np.full(4, 0.3, dtype="<f4"),
        ]
    )
    logical_rgb.tofile(source)

    serialize_vapoursynth_rgbs(source, physical_gbr, shape=(3, 2, 2))
    normalize_vapoursynth_rgbs(physical_gbr, output, shape=(3, 2, 2))

    assert np.array_equal(np.fromfile(output, dtype="<f4"), logical_rgb)


def test_workload_manifests_fix_tensor_quality_contract() -> None:
    for path in Path("benchmarks/workloads").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        quality = value["quality"]["model_space"]

        assert quality["contract_version"] == 3
        assert quality["frame_indices"] == [0, 499, 999]
        assert quality["inference"]["canonical_input"] == "trtvideo-production-rgb-f32"
        assert quality["inference"]["input_acceptance"] == "exact-float32"
        assert set(quality["inference"]["output_thresholds"]) == {
            "p99_abs",
            "rmse",
            "min_psnr_db",
        }
        assert quality["preprocessing"] == {"acceptance": "diagnostic-only"}
