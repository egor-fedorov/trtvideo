from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.quality import preflight_tas as preflight
from benchmarks.scripts.quality.model_space import (
    TensorArtifact,
    TensorThresholds,
    evaluate_metrics,
    write_capture_manifest,
)
from benchmarks.scripts.runtime.environment import sha256_file
from benchmarks.scripts.tuning.contract import load_tuning_contract
from benchmarks.scripts.tuning.rank import TuningEvidenceError, load_tas_preflight
from trtvideo.video.nvcodec.encoder import NvencCbrContract

ROOT = Path(__file__).resolve().parents[4]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file(path: Path, contents: bytes = b"evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _record(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, dict]:
    workload = json.loads((ROOT / "benchmarks/workloads/liveaction_span_madrid.json").read_text())
    workload["quality"]["model_space"]["frame_indices"] = [0]
    workload["quality"]["product_output"]["frame_indices"] = [0]
    workload["clip"]["variants"][0].update(width=2, height=2)
    workload["clip"]["variants"][0]["benchmark_output"].update(width=4, height=4)
    workload["model"]["variants"][0].update(input_width=2, input_height=2)
    manifest = tmp_path / "benchmarks/workloads/span.json"
    _json(manifest, workload)
    _file(tmp_path / workload["clip"]["variants"][0]["path"])
    onnx = _file(tmp_path / workload["model"]["variants"][0]["fp16_path"])
    metadata = json.loads((ROOT / "benchmarks/implementations.json").read_text())
    _json(tmp_path / "benchmarks/implementations.json", metadata)
    digest = hashlib.sha256()
    for name in ("tas_runtime.py", "tas_process.py"):
        path = _file(tmp_path / "benchmarks/scripts/runners" / name, name.encode())
        digest.update(name.encode())
        digest.update(path.read_bytes())
    for name in ("project", "tas"):
        engine = _file(tmp_path / f"models/{name}.engine", name.encode())
        _json(
            Path(f"{engine}.json"),
            {
                "engine_sha256": sha256_file(engine),
                "model_sha256": sha256_file(onnx),
                "input": {"shape": [1, 3, 2, 2]},
                "output": {"shape": [1, 3, 4, 4]},
                "io_precision": "fp32",
                "input_profile": None,
                "adapter_sha256": digest.hexdigest(),
                "tas_revision": metadata["implementations"]["tas"]["source_revision"],
            },
        )
    monkeypatch.setattr(preflight, "_revision", lambda root: "revision")
    monkeypatch.setattr(preflight, "_image_id", lambda ref: "id-" + ref)
    args = argparse.Namespace(
        root=str(tmp_path),
        manifest=str(manifest),
        variant="720p",
        gpu_id=0,
        engine="models/project.engine",
        tas_engine="models/tas.engine",
        output_dir=None,
    )
    identity = preflight.build_preflight_identity(
        root=tmp_path,
        manifest=manifest,
        variant="720p",
        engine=Path(args.engine),
        tas_engine=Path(args.tas_engine),
        gpu_id=0,
    )
    return args, identity


def _gate_files(
    directory: Path, root: Path, identity: dict, profile: dict, mode: str = "valid"
) -> int:
    workload = json.loads((root / identity["files"]["workload_manifest"]["path"]).read_text())
    tensor_dir = directory / "tensor-quality"
    reference = tensor_dir / "production/trtvideo/manifest.json"
    captures: dict[tuple[str, str], Path] = {}
    for stage, implementations in (
        ("production", preflight.PRODUCTS),
        ("inference", ("vstrt", "tas")),
    ):
        for implementation in implementations:
            path = tensor_dir / stage / implementation / "manifest.json"
            scope = (
                "production-reference"
                if implementation == "trtvideo"
                else (
                    "production-preprocessing"
                    if stage == "production"
                    else "shared-input-inference"
                )
            )
            artifacts = []
            for tensor_stage in (
                ("input",) if scope == "production-preprocessing" else ("input", "output")
            ):
                shape = (3, 2, 2) if tensor_stage == "input" else (3, 4, 4)
                contents = struct.pack("<f", 0.5) * math.prod(shape)
                if (
                    mode == "canonical"
                    and implementation == "tas"
                    and tensor_stage == "input"
                    and stage == "inference"
                ):
                    contents = struct.pack("<f", 0.1) * math.prod(shape)
                tensor = _file(path.parent / f"{tensor_stage}.f32", contents)
                artifacts.append(
                    TensorArtifact(
                        stage=tensor_stage,
                        frame_index=0,
                        shape=shape,
                        path=tensor.name,
                        size_bytes=tensor.stat().st_size,
                        sha256=sha256_file(tensor),
                    )
                )
            write_capture_manifest(
                path,
                implementation=preflight.PRODUCTS[implementation],
                capture_scope=scope,
                workload_id=identity["workload_id"],
                variant=identity["variant"],
                input_sha256=identity["files"]["input"]["sha256"],
                onnx_sha256=identity["files"]["onnx"]["sha256"],
                engine_sha256=identity["files"][
                    "tas_engine" if implementation == "tas" else "engine"
                ]["sha256"],
                image={
                    "id": identity["images"][implementation]["id"],
                    "repository_revision": identity["repository_revision"],
                    "source_dirty": "0",
                },
                execution_profile=preflight._expected_profile(implementation, profile),
                artifacts=artifacts,
                canonical_input_manifest_sha256=sha256_file(reference)
                if stage == "inference"
                else None,
            )
            captures[(stage, implementation)] = path
    reference_value = json.loads(reference.read_text())
    thresholds = TensorThresholds.from_dict(
        workload["quality"]["model_space"]["inference"]["output_thresholds"], stage="output"
    )
    for preprocessing in (True, False):
        name = "preprocessing-diagnostic" if preprocessing else "inference-parity"
        report_errors = []
        comparisons = []
        for implementation in ("vstrt", "tas"):
            capture_path = captures[
                ("production" if preprocessing else "inference", implementation)
            ]
            capture = json.loads(capture_path.read_text())
            errors = []
            tensors = []
            for tensor_stage in ("input", "output"):
                metrics = {"finite": True, "exact": True, "p99_abs": 0, "rmse": 0, "psnr_db": None}
                if (
                    not preprocessing
                    and tensor_stage == "output"
                    and (
                        (implementation == "tas" and mode == "inference")
                        or (implementation == "vstrt" and mode == "vs-failure")
                    )
                ):
                    metrics.update(exact=False, p99_abs=0.2, rmse=0.1, psnr_db=20)
                tensor_errors = evaluate_metrics(metrics, thresholds)
                errors.extend(f"{tensor_stage} frame 0: {error}" for error in tensor_errors)
                tensors.append(
                    {
                        "stage": tensor_stage,
                        "frame_index": 0,
                        "metrics": metrics,
                        "status": "invalid" if tensor_errors else "valid",
                        "errors": tensor_errors,
                    }
                )
            comparisons.append(
                {
                    "implementation": preflight.PRODUCTS[implementation],
                    "capture_manifest_sha256": sha256_file(capture_path),
                    "engine_sha256": capture["assets"]["engine_sha256"],
                    "image": capture["environment"]["image"],
                    "execution_profile": capture["execution_profile"],
                    "canonical_input_manifest_sha256": sha256_file(reference),
                    "status": "complete" if preprocessing else ("invalid" if errors else "valid"),
                    "errors": errors,
                    "tensors": tensors,
                }
            )
            report_errors.extend(
                f"{preflight.PRODUCTS[implementation]}: {error}" for error in errors
            )
        _json(
            tensor_dir / f"{name}.json",
            {
                "schema_version": 1,
                "document_type": name,
                "acceptance_gate": not preprocessing,
                "status": "complete"
                if preprocessing
                else ("invalid" if report_errors else "valid"),
                "publishable": not report_errors,
                "execution_profile": "tuned",
                "contract_version": 3,
                "workload_id": identity["workload_id"],
                "variant": identity["variant"],
                "frame_indices": [0],
                "reference": {
                    "capture_manifest_sha256": sha256_file(reference),
                    "engine_sha256": reference_value["assets"]["engine_sha256"],
                },
                "assets": {
                    "input_sha256": identity["files"]["input"]["sha256"],
                    "onnx_sha256": identity["files"]["onnx"]["sha256"],
                    "canonical_input_manifest_sha256": sha256_file(reference),
                },
                "comparisons": comparisons,
                "errors": report_errors,
                "thresholds": {"output": thresholds.as_dict()},
            },
        )
    if mode in {"inference", "vs-failure", "canonical"}:
        return 2
    product_dir = directory / "product-output"
    comparisons = []
    crops = {}
    errors = []
    for implementation, product in preflight.PRODUCTS.items():
        path = product_dir / implementation / "run-01/manifest.json"
        output = _file(path.parent / "output.mp4")
        engine_hash = identity["files"]["tas_engine" if implementation == "tas" else "engine"][
            "sha256"
        ]
        _json(
            path,
            {
                "status": "invalid" if mode == "thermal" and implementation == "tas" else "valid",
                "errors": ["Invalid throttle reasons observed: sw_thermal_slowdown"]
                if mode == "thermal"
                else [],
                "product": product,
                "workload_id": identity["workload_id"],
                "variant": identity["variant"],
                "parameters": {
                    "frames": 1000,
                    "encoder": NvencCbrContract(bitrate_bps=35_000_000, gop_frames=24).as_dict(),
                    **preflight._expected_profile(implementation, profile),
                },
                "assets": {
                    "input": {"sha256": identity["files"]["input"]["sha256"]},
                    "onnx": {"sha256": identity["files"]["onnx"]["sha256"]},
                    "engine": {"sha256": engine_hash},
                },
                "environment": {
                    "image": {
                        "id": identity["images"][implementation]["id"],
                        "source_dirty": "0",
                        "repository_revision": identity["repository_revision"],
                    }
                },
                "reproducibility": {"publishable": True},
                "measured": {"validation": {"valid": True}, "output": _record(output, root)},
            },
        )
        entry = {
            "implementation": product,
            "engine_sha256": engine_hash,
            "output_sha256": sha256_file(output),
            "run_manifest": str(path.relative_to(root)),
            "run_manifest_sha256": sha256_file(path),
        }
        if implementation == "trtvideo":
            product_reference = entry
        else:
            metrics = {}
            bad = mode == "product" and implementation == "tas"
            for name in ("psnr", "ssim"):
                stats = _file(path.parent / f"{name}.stats")
                log = _file(path.parent / f"{name}.log")
                metrics[name] = {
                    "stats_path": str(stats.relative_to(root)),
                    "stats_sha256": sha256_file(stats),
                    "ffmpeg_log": str(log.relative_to(root)),
                    "ffmpeg_log_sha256": sha256_file(log),
                }
            metrics["psnr"].update(exact=False, average_db=30 if bad else 50)
            metrics["ssim"]["all"] = 0.8 if bad else 0.999
            errors = (
                ["PSNR must be >= 35 dB, got 30 dB", "SSIM must be >= 0.95, got 0.8"] if bad else []
            )
            comparisons.append(
                {
                    **entry,
                    "metrics": metrics,
                    "status": "invalid" if errors else "valid",
                    "errors": errors,
                }
            )
        crops[product] = []
        for crop in workload["quality"]["product_output"]["crops"]:
            crop_path = _file(path.parent / f"{crop['name']}.png")
            crops[product].append(
                {**_record(crop_path, root), "frame_index": 0, "crop": crop["name"]}
            )
    _json(
        product_dir / "product-output-parity.json",
        {
            "schema_version": 1,
            "document_type": "product-output-parity",
            "workload_id": identity["workload_id"],
            "variant": identity["variant"],
            "frame_indices": [0],
            "status": "invalid" if errors else "valid",
            "publishable": not errors,
            "thresholds": workload["quality"]["product_output"]["thresholds"],
            "assets": {
                "input_sha256": identity["files"]["input"]["sha256"],
                "onnx_sha256": identity["files"]["onnx"]["sha256"],
            },
            "reference": product_reference,
            "comparisons": comparisons,
            "visual_crops": crops,
            "errors": [f"TheAnimeScripter: {error}" for error in errors],
        },
    )
    return 2 if mode != "valid" else 0


def _runner(
    monkeypatch: pytest.MonkeyPatch, context: tuple, modes: dict | None = None
) -> list[str]:
    args, identity = context
    root = Path(args.root)
    calls = []

    def run(command: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess:
        assert command[:4] == ["make", "-C", str(root / "benchmarks"), "quality-gates"]
        assert "-j1" in command
        variables = dict(value.split("=", 1) for value in command[4:] if "=" in value)
        assert variables["VSTRT_ARGS"] == preflight.VSTRT_ARGS
        assert variables["EXECUTION_PROFILE"] == "tuned"
        directory = (root / variables["MODEL_SPACE_DIR"]).parent
        candidate_id = directory.name
        calls.append(candidate_id)
        profile = preflight.PROFILES[candidate_id]
        assert (
            variables["TAS_ARGS"]
            == f"--decode-method {profile['decode_method']} --writer {profile['writer']}"
        )
        code = _gate_files(
            directory, root, identity, profile, (modes or {}).get(candidate_id, "valid")
        )
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(preflight.subprocess, "run", run)
    return calls


def _report_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.root)
        / "artefacts/benchmarks/comparative/tuning/span-720p/tas-preflight/preflight.json"
    )


def test_all_four_profiles_have_immutable_hashed_evidence(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, identity = context
    calls = _runner(monkeypatch, context)
    report = preflight.run_preflight(args)
    assert calls == list(preflight.PROFILES)
    assert report["status"] == "valid"
    assert report["eligible_candidates"] == list(preflight.PROFILES)
    assert report["identity"] == identity
    assert preflight.load_preflight_report(_report_path(args), root=Path(args.root)) == report
    calls.clear()
    original = _report_path(args).read_bytes()
    assert preflight.run_preflight(args) == report
    assert calls == []
    assert _report_path(args).read_bytes() == original


@pytest.mark.parametrize("mode", ["inference", "product"])
def test_quality_failure_disqualifies_only_tas_configuration(
    context: tuple, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    args, _ = context
    calls = _runner(monkeypatch, context, {"tas-cpu-ffmpeg": mode})
    report = preflight.run_preflight(args)
    assert len(calls) == 4
    assert report["status"] == "valid"
    assert report["profiles"]["tas-cpu-ffmpeg"]["status"] == "disqualified"
    assert report["eligible_candidates"] == list(preflight.PROFILES)[1:]
    assert preflight.load_preflight_report(_report_path(args), root=Path(args.root)) == report


@pytest.mark.parametrize(
    "mode,match",
    [
        ("thermal", "sw_thermal_slowdown"),
        ("canonical", "Shared-input tensor"),
        ("vs-failure", "vs-mlrt inference"),
    ],
)
def test_infrastructure_failure_cannot_be_disqualified(
    context: tuple, monkeypatch: pytest.MonkeyPatch, mode: str, match: str
) -> None:
    args, _ = context
    calls = _runner(monkeypatch, context, {"tas-cpu-ffmpeg": mode})
    with pytest.raises(preflight.PreflightError, match=match):
        preflight.run_preflight(args)
    assert calls == ["tas-cpu-ffmpeg"]
    assert not _report_path(args).exists()
    assert not (_report_path(args).parent / "tas-cpu-ffmpeg/result.json").exists()


def test_no_eligible_profile_is_reported_without_abandoning_grid(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    calls = _runner(monkeypatch, context, {key: "inference" for key in preflight.PROFILES})
    report = preflight.run_preflight(args)
    assert len(calls) == 4
    assert report["status"] == "invalid"
    assert report["eligible_candidates"] == []
    monkeypatch.setattr(
        preflight, "build_parser", lambda: type("Parser", (), {"parse_args": lambda self: args})()
    )
    with pytest.raises(SystemExit) as caught:
        preflight.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "asset", ["onnx", "input", "adapter_runtime", "engine", "tas_engine", "workload_manifest"]
)
def test_stale_source_or_asset_prevents_preflight_reuse(
    context: tuple, monkeypatch: pytest.MonkeyPatch, asset: str
) -> None:
    args, identity = context
    _runner(monkeypatch, context)
    preflight.run_preflight(args)
    (Path(args.root) / identity["files"][asset]["path"]).write_bytes(b"changed")
    with pytest.raises(preflight.PreflightError, match="SHA256 changed"):
        preflight.load_preflight_report(_report_path(args), root=Path(args.root))


def test_retained_evidence_tamper_prevents_reuse(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    calls = _runner(monkeypatch, context)
    preflight.run_preflight(args)
    calls.clear()
    tensor = _report_path(args).parent / "tas-cpu-ffmpeg/tensor-quality/inference/tas/output.f32"
    tensor.write_bytes(b"changed")
    with pytest.raises(preflight.PreflightError, match="SHA256 changed"):
        preflight.run_preflight(args)
    assert not calls


def test_retagged_image_prevents_reuse(context: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _ = context
    _runner(monkeypatch, context)
    preflight.run_preflight(args)
    monkeypatch.setattr(preflight, "_image_id", lambda reference: "rebuilt-image")
    with pytest.raises(preflight.PreflightError, match="Docker image changed"):
        preflight.load_preflight_report(_report_path(args), root=Path(args.root))


def test_missing_gate_is_infrastructure_failure(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2),
    )
    with pytest.raises(RuntimeError, match="Cannot read JSON"):
        preflight.run_preflight(args)
    assert not _report_path(args).exists()


def test_completed_profile_resumes_but_partial_profile_is_not_overwritten(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    _runner(monkeypatch, context, {"tas-cpu-nelux": "thermal"})
    with pytest.raises(preflight.PreflightError):
        preflight.run_preflight(args)
    completed = _report_path(args).parent / "tas-cpu-ffmpeg/result.json"
    original = completed.read_bytes()
    calls = _runner(monkeypatch, context)
    with pytest.raises(preflight.PreflightError, match="Partial preflight evidence"):
        preflight.run_preflight(args)
    assert completed.read_bytes() == original
    assert calls == []
    partial = _report_path(args).parent / "tas-cpu-nelux"
    partial.rename(Path(args.root) / "failed-profile")
    report = preflight.run_preflight(args)
    assert calls == list(preflight.PROFILES)[1:]
    assert report["status"] == "valid"


def test_external_expected_identity_must_match(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, identity = context
    _runner(monkeypatch, context)
    preflight.run_preflight(args)
    expected = copy.deepcopy(identity)
    expected["gpu_id"] = 1
    with pytest.raises(preflight.PreflightError, match="Stale TAS preflight identity"):
        preflight.load_preflight_report(
            _report_path(args), root=Path(args.root), expected_identity=expected
        )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("id", "other-image", "capture image changed"),
        ("repository_revision", "other-revision", "capture revision changed"),
        ("source_dirty", "1", "not publishable"),
    ],
)
def test_tas_capture_must_bind_clean_image_and_revision(
    context: tuple, monkeypatch: pytest.MonkeyPatch, key: str, value: str, match: str
) -> None:
    args, _ = context
    original = _gate_files

    def changed(directory: Path, *arguments: Any) -> int:
        code = original(directory, *arguments)
        path = directory / "tensor-quality/production/tas/manifest.json"
        capture = json.loads(path.read_text())
        capture["environment"]["image"][key] = value
        _json(path, capture)
        return code

    monkeypatch.setitem(globals(), "_gate_files", changed)
    _runner(monkeypatch, context)
    with pytest.raises(RuntimeError, match=match):
        preflight.run_preflight(args)
    assert not _report_path(args).exists()


def test_nvml_failure_is_fatal_even_when_manifest_status_says_valid(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    original = _gate_files

    def changed(directory: Path, *arguments: Any) -> int:
        code = original(directory, *arguments)
        path = directory / "product-output/tas/run-01/manifest.json"
        manifest = json.loads(path.read_text())
        manifest.setdefault("measured", {})["metrics"] = {
            "nvml": {"valid": False, "errors": ["sw_thermal_slowdown"]}
        }
        _json(path, manifest)
        return code

    monkeypatch.setitem(globals(), "_gate_files", changed)
    _runner(monkeypatch, context)
    with pytest.raises(preflight.PreflightError, match="sw_thermal_slowdown"):
        preflight.run_preflight(args)
    assert not _report_path(args).exists()


def test_valid_reports_cannot_hide_failed_quality_command(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    _runner(monkeypatch, context)
    original = preflight.subprocess.run

    def failed(*arguments: Any, **options: Any) -> subprocess.CompletedProcess:
        result = original(*arguments, **options)
        result.returncode = 2
        return result

    monkeypatch.setattr(preflight.subprocess, "run", failed)
    with pytest.raises(preflight.PreflightError, match="exit code contradicts"):
        preflight.run_preflight(args)
    assert not _report_path(args).exists()


def test_output_directory_must_be_visible_to_quality_containers(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = context
    args.output_dir = "unmounted-output"
    calls = _runner(monkeypatch, context)
    with pytest.raises(preflight.PreflightError, match="artefacts"):
        preflight.run_preflight(args)
    assert calls == []


def test_preflight_syntax_supports_host_python_310() -> None:
    ast.parse(Path(preflight.__file__).read_text(), feature_version=(3, 10))


def test_rank_consumes_real_verified_preflight_without_reclassifying_failures(
    context: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, identity = context
    _runner(monkeypatch, context, {"tas-cpu-ffmpeg": "inference"})
    report = preflight.run_preflight(args)
    path = _report_path(args)
    options = {
        "root": Path(args.root),
        "contract": load_tuning_contract(ROOT / "benchmarks/tuning/span_candidates.json"),
        "workload": json.loads(Path(args.manifest).read_text()),
        "variant": args.variant,
        "engine_sha256": identity["files"]["engine"]["sha256"],
        "tas_engine_sha256": identity["files"]["tas_engine"]["sha256"],
        "workload_sha256": identity["files"]["workload_manifest"]["sha256"],
    }
    selected = load_tas_preflight(path, **options)
    assert selected["tas-cpu-ffmpeg"]["status"] == "invalid"
    assert [key for key, value in selected.items() if value["status"] == "valid"] == report[
        "eligible_candidates"
    ]
    for candidate_id, value in selected.items():
        assert (
            value["quality"]["inference_parity"]
            == report["profiles"][candidate_id]["evidence"]["inference"]
        )
    evidence = path.parent / "tas-cpu-nelux/tensor-quality/production/tas/input.f32"
    evidence.write_bytes(b"changed")
    with pytest.raises(TuningEvidenceError, match="preflight evidence is invalid"):
        load_tas_preflight(path, **options)
