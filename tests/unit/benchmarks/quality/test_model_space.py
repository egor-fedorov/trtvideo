from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.scripts.quality.capture_vspipe import (
    build_capture_command,
    normalize_vapoursynth_rgbs,
)
from benchmarks.scripts.quality.model_space import (
    CaptureManifest,
    ModelSpaceError,
    TensorThresholds,
    compare_captures,
    create_tensor_artifact,
    evaluate_metrics,
    parse_frame_indices,
    write_capture_manifest,
)


def _write_capture(
    root: Path,
    *,
    implementation: str,
    engine_sha256: str,
    input_value: float,
    output_value: float,
    execution_profile: dict[str, object] | None = None,
) -> Path:
    root.mkdir(parents=True)
    artifacts = []
    for stage, value in (("input", input_value), ("output", output_value)):
        path = root / f"{stage}.frame-000000.f32"
        np.full((3, 2, 2), value, dtype="<f4").tofile(path)
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
    )
    return manifest_path


def _thresholds(limit: float = 0.01) -> dict[str, TensorThresholds]:
    return {
        stage: TensorThresholds(
            p99_abs=limit,
            rmse=limit,
            min_psnr_db=40.0,
        )
        for stage in ("input", "output")
    }


def test_parse_frame_indices_sorts_and_deduplicates() -> None:
    assert parse_frame_indices("999,0,499,499", frame_count=1000) == (0, 499, 999)


def test_parse_frame_indices_rejects_out_of_range() -> None:
    with pytest.raises(ModelSpaceError, match="must stay"):
        parse_frame_indices("0,1000", frame_count=1000)


def test_capture_manifest_detects_changed_tensor(tmp_path: Path) -> None:
    manifest_path = _write_capture(
        tmp_path / "capture",
        implementation="reference",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    tensor_path = manifest_path.parent / "input.frame-000000.f32"
    tensor_path.write_bytes(b"\0" * tensor_path.stat().st_size)

    with pytest.raises(ModelSpaceError, match="SHA256 changed"):
        CaptureManifest.load(manifest_path)


def test_compare_captures_accepts_close_float_tensors(tmp_path: Path) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        engine_sha256="3" * 64,
        input_value=0.2501,
        output_value=0.5001,
    )

    report = compare_captures(
        reference,
        [candidate],
        thresholds=_thresholds(),
    )

    assert report["status"] == "valid"
    assert report["publishable"] is True
    assert report["execution_profile"] == "upstream-default"
    assert report["frame_indices"] == [0]
    assert report["comparisons"][0]["status"] == "valid"
    assert "comparison_class" not in report["comparisons"][0]


def test_compare_captures_rejects_large_difference(tmp_path: Path) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="VSGAN-tensorrt-docker",
        engine_sha256="4" * 64,
        input_value=0.25,
        output_value=0.75,
    )

    report = compare_captures(
        reference,
        [candidate],
        thresholds=_thresholds(),
    )

    assert report["status"] == "invalid"
    assert any("output frame 0" in error for error in report["errors"])


def test_compare_captures_rejects_mixed_execution_profiles(
    tmp_path: Path,
) -> None:
    reference = _write_capture(
        tmp_path / "reference",
        implementation="trtvideo",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
    )
    candidate = _write_capture(
        tmp_path / "candidate",
        implementation="vs-mlrt",
        engine_sha256="3" * 64,
        input_value=0.25,
        output_value=0.5,
        execution_profile={
            "execution_profile": "tuned",
            "vspipe_requests": "auto",
            "num_streams": 3,
            "vapoursynth_threads": 6,
            "cuda_graph": True,
        },
    )

    report = compare_captures(
        reference,
        [candidate],
        thresholds=_thresholds(),
    )

    assert report["status"] == "invalid"
    assert any("execution profile differs" in error for error in report["errors"])


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


def test_vspipe_capture_command_outputs_raw_rgbs(tmp_path: Path) -> None:
    args = argparse.Namespace(
        implementation="vsgan",
        engine="/app/models/model.engine",
        gpu_id=0,
        execution_profile="upstream-default",
        requests=None,
        num_streams=None,
        vs_threads=None,
        cuda_graph=None,
        script="/app/benchmarks/vsgan/upscale.vpy",
    )

    command = build_capture_command(
        args,
        input_path=Path("/app/videos/input.mp4"),
        output_path=tmp_path / "output.f32",
        frame_index=499,
        stage="output",
    )

    assert "--container" not in command
    assert command[command.index("--start") + 1] == "499"
    assert "model_space_stage=output" in command
    assert "vs_threads=4" in command
    assert command[-1] == str(tmp_path / "output.f32")


def test_vspipe_capture_command_uses_tuned_execution_profile(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        implementation="vstrt",
        engine="/app/models/model.engine",
        gpu_id=0,
        execution_profile="tuned",
        requests="auto",
        num_streams=3,
        vs_threads=6,
        cuda_graph=True,
        script="/app/benchmarks/vstrt/upscale.vpy",
    )

    command = build_capture_command(
        args,
        input_path=Path("/app/videos/input.mp4"),
        output_path=tmp_path / "output.f32",
        frame_index=499,
        stage="output",
    )

    assert "--requests" not in command
    assert "num_streams=3" in command
    assert "cuda_graph=1" in command
    assert "vs_threads=6" in command


def test_vapoursynth_gbr_planes_are_normalized_to_rgb(tmp_path: Path) -> None:
    source = tmp_path / "source.f32"
    output = tmp_path / "output.f32"
    np.concatenate(
        [
            np.full(4, 0.2, dtype="<f4"),
            np.full(4, 0.3, dtype="<f4"),
            np.full(4, 0.1, dtype="<f4"),
        ]
    ).tofile(source)

    normalize_vapoursynth_rgbs(source, output, shape=(3, 2, 2))

    normalized = np.fromfile(output, dtype="<f4").reshape(3, 2, 2)
    assert np.all(normalized[0] == np.float32(0.1))
    assert np.all(normalized[1] == np.float32(0.2))
    assert np.all(normalized[2] == np.float32(0.3))


def test_workload_manifests_fix_model_space_contract() -> None:
    for path in (
        Path("benchmarks/workloads/realesrgan_x2plus_sintel.json"),
        Path("benchmarks/workloads/liveaction_span_sintel.json"),
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        quality = value["quality"]["model_space"]

        assert quality["frame_indices"] == [0, 499, 999]
        assert set(quality["thresholds"]) == {"input", "output"}
        assert set(quality["thresholds"]["input"]) == {
            "p99_abs",
            "rmse",
            "min_psnr_db",
        }
