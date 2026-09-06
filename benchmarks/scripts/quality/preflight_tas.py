"""Preflight all TAS I/O paths before spending GPU time on tuning (host Python 3.10)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.contracts.engine import (
    load_engine_contract,
    validate_static_engine_contract,
)
from benchmarks.scripts.contracts.manifest import artifact_path, execution_profile, load_json
from benchmarks.scripts.quality.model_space import (
    CaptureManifest,
    TensorThresholds,
    evaluate_metrics,
)
from benchmarks.scripts.quality.product_output import OutputEvidence
from benchmarks.scripts.runtime.environment import sha256_file
from benchmarks.scripts.workloads.manifest import (
    find_clip_variant,
    find_model_variant,
    load_manifest,
)
from trtvideo.video.nvcodec.encoder import NvencCbrContract, gop_size_for_one_second

PROFILES = {
    f"tas-{decode}-{writer}": {
        "execution_profile": "tuned",
        "decode_method": decode,
        "writer": writer,
        "cuda_graph": True,
    }
    for decode in ("cpu", "nvdec")
    for writer in ("ffmpeg", "nelux")
}
VSTRT_PROFILE = {
    "execution_profile": "tuned",
    "vspipe_requests": "auto",
    "num_streams": 1,
    "vapoursynth_threads": "auto",
    "cuda_graph": False,
}
VSTRT_ARGS = "--requests auto --num-streams 1 --vs-threads auto --no-cuda-graph"
PRODUCTS = {"trtvideo": "trtvideo", "vstrt": "vs-mlrt", "tas": "TheAnimeScripter"}


class PreflightError(RuntimeError):
    """Infrastructure or stale evidence must stop the search, not disqualify a model."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _path(root: Path, value: str | Path) -> Path:
    return artifact_path(root, str(value), label="Preflight path")


def _record(path: Path, root: Path) -> dict[str, str]:
    path = _path(root, path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _verify_record(record: dict[str, Any], root: Path) -> Path:
    path = _path(root, record["path"])
    _require(sha256_file(path) == record["sha256"], f"Preflight evidence SHA256 changed: {path}")
    return path


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation leaves old evidence intact, including after an interrupted write.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _image_id(reference: str) -> str:
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(bool(image_id), f"Docker image identity is missing: {reference}")
    return image_id


def build_preflight_identity(
    *, root: Path, manifest: Path, variant: str, engine: Path, tas_engine: Path, gpu_id: int
) -> dict[str, Any]:
    """Resolve the exact source/assets/images against which preflight may be reused."""
    root = root.resolve()
    _require(gpu_id >= 0, "GPU id must be non-negative")
    manifest = _path(root, manifest)
    workload = load_manifest(manifest)
    metadata_path = root / "benchmarks/implementations.json"
    metadata = load_json(metadata_path)["implementations"]
    onnx = _path(root, find_model_variant(workload, variant)["fp16_path"])
    files = {
        "workload_manifest": manifest,
        "input": _path(root, find_clip_variant(workload, variant)["path"]),
        "onnx": onnx,
        "engine": _path(root, engine),
        "tas_engine": _path(root, tas_engine),
        "implementations": metadata_path,
        "adapter_runtime": root / "benchmarks/scripts/runners/tas_runtime.py",
        "adapter_process": root / "benchmarks/scripts/runners/tas_process.py",
    }
    adapter = hashlib.sha256()
    for name in ("adapter_runtime", "adapter_process"):
        adapter.update(files[name].name.encode("ascii"))
        adapter.update(files[name].read_bytes())
    for name in ("engine", "tas_engine"):
        sidecar, sidecar_path = load_engine_contract(files[name])
        validate_static_engine_contract(sidecar, workload, variant, onnx)
        files[f"{name}_manifest"] = sidecar_path
        if name == "tas_engine":
            _require(
                sidecar.get("adapter_sha256") == adapter.hexdigest(), "TAS engine adapter changed"
            )
            _require(
                sidecar.get("tas_revision") == metadata["tas"]["source_revision"],
                "TAS engine upstream revision changed",
            )
    references = {
        "trtvideo": metadata["trtexec"]["image"],
        "vstrt": metadata["vstrt"]["image"],
        "tas": metadata["tas"]["image"],
    }
    return {
        "workload_id": workload["id"],
        "variant": variant,
        "gpu_id": gpu_id,
        "repository_revision": _revision(root),
        "tas_revision": metadata["tas"]["source_revision"],
        "adapter_sha256": adapter.hexdigest(),
        "files": {key: _record(path, root) for key, path in files.items()},
        "images": {
            key: {"reference": ref, "id": _image_id(ref)} for key, ref in references.items()
        },
    }


def _expected_profile(implementation: str, profile: dict[str, Any]) -> dict[str, Any]:
    if implementation == "trtvideo":
        return {"execution_profile": "tuned", "cuda_graph": False}
    return VSTRT_PROFILE if implementation == "vstrt" else profile


def _capture(
    path: Path,
    *,
    implementation: str,
    identity: dict[str, Any],
    profile: dict[str, Any],
    scope: str,
) -> CaptureManifest:
    capture = CaptureManifest.load(path)
    if implementation == "tas":
        execution_profile(capture.execution_profile)
    expected_engine = "tas_engine" if implementation == "tas" else "engine"
    checks = {
        "implementation": (capture.implementation, PRODUCTS[implementation]),
        "scope": (capture.capture_scope, scope),
        "workload": (capture.workload_id, identity["workload_id"]),
        "variant": (capture.variant, identity["variant"]),
        "input": (capture.input_sha256, identity["files"]["input"]["sha256"]),
        "ONNX": (capture.onnx_sha256, identity["files"]["onnx"]["sha256"]),
        "engine": (capture.engine_sha256, identity["files"][expected_engine]["sha256"]),
        "image": (capture.image["id"], identity["images"][implementation]["id"]),
        "revision": (capture.image["repository_revision"], identity["repository_revision"]),
        "profile": (capture.execution_profile, _expected_profile(implementation, profile)),
    }
    for label, (actual, expected) in checks.items():
        _require(actual == expected, f"{path}: capture {label} changed")
    return capture


def _comparison_set(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparisons = report["comparisons"]
    _require(isinstance(comparisons, list), "Quality comparisons must be an array")
    result = {item["implementation"]: item for item in comparisons}
    _require(
        len(comparisons) == 2 and set(result) == {"vs-mlrt", "TheAnimeScripter"},
        "Quality comparison set changed",
    )
    return result


def _report_scope(
    report: dict[str, Any], identity: dict[str, Any], workload: dict[str, Any], *, tensor: bool
) -> None:
    checks = {
        "workload_id": identity["workload_id"],
        "variant": identity["variant"],
        "frame_indices": workload["quality"]["model_space" if tensor else "product_output"][
            "frame_indices"
        ],
    }
    for key, expected in checks.items():
        _require(report.get(key) == expected, f"Quality report changed {key}")
    for name in ("input", "onnx"):
        _require(
            report["assets"][f"{name}_sha256"] == identity["files"][name]["sha256"],
            f"Quality report changed {name} SHA256",
        )


def _tensor_gate(
    directory: Path,
    identity: dict[str, Any],
    profile: dict[str, Any],
    workload: dict[str, Any],
    *,
    preprocessing: bool,
) -> list[str]:
    name = "preprocessing-diagnostic" if preprocessing else "inference-parity"
    report = load_json(directory / f"{name}.json")
    _require(report.get("document_type") == name, f"Wrong {name} document type")
    _report_scope(report, identity, workload, tensor=True)
    _require(report.get("execution_profile") == "tuned", "Tensor report profile changed")
    _require(
        report.get("contract_version") == workload["quality"]["model_space"]["contract_version"],
        "Tensor report contract changed",
    )
    _require(report.get("acceptance_gate") is (not preprocessing), "Tensor acceptance role changed")
    reference_path = directory / "production/trtvideo/manifest.json"
    reference = _capture(
        reference_path,
        implementation="trtvideo",
        identity=identity,
        profile=profile,
        scope="production-reference",
    )
    model = find_model_variant(workload, identity["variant"])
    output = find_clip_variant(workload, identity["variant"])["benchmark_output"]
    shapes = {
        "input": (3, model["input_height"], model["input_width"]),
        "output": (3, output["height"], output["width"]),
    }
    indices = workload["quality"]["model_space"]["frame_indices"]
    _require(
        set(reference.artifact_map()) == {(stage, index) for stage in shapes for index in indices}
        and all(item.shape == shapes[item.stage] for item in reference.artifacts),
        "Reference tensor shape/frame set differs from workload",
    )
    _require(
        report["reference"]["capture_manifest_sha256"] == sha256_file(reference_path),
        "Reference capture changed",
    )
    comparisons = _comparison_set(report)
    failures: list[str] = []
    for implementation in ("vstrt", "tas"):
        path = (
            directory
            / ("production" if preprocessing else "inference")
            / implementation
            / "manifest.json"
        )
        capture = _capture(
            path,
            implementation=implementation,
            identity=identity,
            profile=profile,
            scope="production-preprocessing" if preprocessing else "shared-input-inference",
        )
        comparison = comparisons[PRODUCTS[implementation]]
        for key, value in {
            "capture_manifest_sha256": sha256_file(path),
            "engine_sha256": capture.engine_sha256,
            "execution_profile": capture.execution_profile,
            "image": capture.image,
        }.items():
            _require(comparison.get(key) == value, f"{name}: {implementation} {key} changed")
        if preprocessing:
            _require(
                set(capture.artifact_map()) == {("input", index) for index in indices}
                and all(item.shape == shapes["input"] for item in capture.artifacts),
                "Preprocessing tensor set changed",
            )
            _require(
                comparison.get("status") == "complete" and not comparison.get("errors"),
                "Preprocessing infrastructure failure",
            )
            continue
        reference_hash = sha256_file(reference_path)
        _require(
            capture.canonical_input_manifest_sha256 == reference_hash
            and comparison.get("canonical_input_manifest_sha256") == reference_hash
            and report["assets"].get("canonical_input_manifest_sha256") == reference_hash,
            "Shared-input source changed",
        )
        reference_tensors = reference.artifact_map()
        candidate_tensors = capture.artifact_map()
        _require(set(reference_tensors) == set(candidate_tensors), "Tensor set changed")
        tensors = comparison["tensors"]
        _require(
            len(tensors) == len(reference_tensors)
            and {(item["stage"], item["frame_index"]) for item in tensors}
            == set(reference_tensors),
            "Inference tensor evidence is incomplete",
        )
        thresholds = TensorThresholds.from_dict(
            workload["quality"]["model_space"]["inference"]["output_thresholds"], stage="output"
        )
        _require(
            report.get("thresholds") == {"output": thresholds.as_dict()},
            "Inference thresholds changed",
        )
        errors: list[str] = []
        for tensor in tensors:
            tensor_key = (tensor["stage"], tensor["frame_index"])
            _require(
                candidate_tensors[tensor_key].shape == reference_tensors[tensor_key].shape,
                "Tensor shape changed",
            )
            if tensor_key[0] == "input":
                _require(
                    candidate_tensors[tensor_key].sha256 == reference_tensors[tensor_key].sha256,
                    "Shared-input tensor differs",
                )
                expected_errors: list[str] = []
            else:
                expected_errors = evaluate_metrics(tensor["metrics"], thresholds)
            _require(
                tensor.get("errors") == expected_errors,
                "Inference tensor errors contradict metrics",
            )
            _require(
                tensor.get("status") == ("invalid" if expected_errors else "valid"),
                "Inference tensor status contradicts metrics",
            )
            errors.extend(
                f"{tensor_key[0]} frame {tensor_key[1]}: {error}" for error in expected_errors
            )
        _require(
            comparison.get("errors") == errors, "Inference errors include an infrastructure failure"
        )
        _require(
            comparison.get("status") == ("invalid" if errors else "valid"),
            "Inference comparison status changed",
        )
        _require(not errors or implementation == "tas", "vs-mlrt inference control failed")
        failures.extend(errors)
    expected_status = "complete" if preprocessing else ("invalid" if failures else "valid")
    _require(report.get("status") == expected_status, f"{name} status contradicts comparisons")
    _require(report.get("publishable") is (not failures), f"{name} publication status changed")
    _require(
        report.get("errors") == [f"TheAnimeScripter: {error}" for error in failures],
        f"{name} has unaccounted errors",
    )
    return failures


def _product_gate(
    directory: Path,
    root: Path,
    identity: dict[str, Any],
    profile: dict[str, Any],
    workload: dict[str, Any],
) -> list[str]:
    report = load_json(directory / "product-output-parity.json")
    _require(report.get("document_type") == "product-output-parity", "Wrong product gate document")
    _report_scope(report, identity, workload, tensor=False)
    thresholds = workload["quality"]["product_output"]["thresholds"]
    _require(report.get("thresholds") == thresholds, "Product-output thresholds changed")
    comparisons = _comparison_set(report)
    output = find_clip_variant(workload, identity["variant"])["benchmark_output"]
    encoder = NvencCbrContract(
        bitrate_bps=int(output["bitrate_mbps"] * 1_000_000),
        gop_frames=gop_size_for_one_second(workload["clip"]["fps"]),
    ).as_dict()
    failures: list[str] = []
    for implementation, product in PRODUCTS.items():
        path = directory / implementation / "run-01/manifest.json"
        evidence = OutputEvidence.load(path, root=root)
        _require(evidence.encoder == encoder, "Product-output encoder contract changed")
        manifest = load_json(path)
        image = manifest["environment"]["image"]
        _require(
            image.get("id") == identity["images"][implementation]["id"]
            and image.get("repository_revision") == identity["repository_revision"]
            and str(image.get("source_dirty")) == "0",
            "Product-output source/image identity changed",
        )
        if implementation != "trtvideo":
            _require(
                execution_profile(manifest["parameters"])
                == _expected_profile(implementation, profile),
                "Product-output execution profile changed",
            )
        expected_engine = "tas_engine" if implementation == "tas" else "engine"
        _require(
            evidence.product == product
            and evidence.workload_id == identity["workload_id"]
            and evidence.variant == identity["variant"]
            and evidence.frames == workload["clip"]["frames"]
            and evidence.input_sha256 == identity["files"]["input"]["sha256"]
            and evidence.onnx_sha256 == identity["files"]["onnx"]["sha256"]
            and evidence.engine_sha256 == identity["files"][expected_engine]["sha256"],
            "Product-output run identity changed",
        )
        entry = report["reference"] if implementation == "trtvideo" else comparisons[product]
        _require(
            _verify_record(
                {"path": entry["run_manifest"], "sha256": entry["run_manifest_sha256"]}, root
            )
            == path,
            "Product gate references another run",
        )
        _require(
            entry["engine_sha256"] == evidence.engine_sha256
            and entry["output_sha256"] == evidence.output_sha256,
            "Product gate artifact identity changed",
        )
        if implementation == "trtvideo":
            continue
        metrics = entry["metrics"]
        for metric in metrics.values():
            for path_key, hash_key in (
                ("stats_path", "stats_sha256"),
                ("ffmpeg_log", "ffmpeg_log_sha256"),
            ):
                _verify_record({"path": metric[path_key], "sha256": metric[hash_key]}, root)
        psnr = (
            math.inf
            if metrics["psnr"].get("exact") is True
            else float(metrics["psnr"]["average_db"])
        )
        ssim = float(metrics["ssim"]["all"])
        _require(
            not math.isnan(psnr) and math.isfinite(ssim), "Product quality metrics are not finite"
        )
        errors = []
        if psnr < thresholds["psnr_min_db"]:
            errors.append(f"PSNR must be >= {thresholds['psnr_min_db']:g} dB, got {psnr:g} dB")
        if ssim < thresholds["ssim_min"]:
            errors.append(f"SSIM must be >= {thresholds['ssim_min']:g}, got {ssim:g}")
        _require(
            entry.get("errors") == errors
            and entry.get("status") == ("invalid" if errors else "valid"),
            "Product-output errors/status contradict metrics",
        )
        _require(not errors or implementation == "tas", "vs-mlrt product-output control failed")
        failures.extend(errors)
    crops = report["visual_crops"]
    _require(set(crops) == set(PRODUCTS.values()), "Product-output visual crop set changed")
    crop_keys = {
        (index, crop["name"])
        for index in workload["quality"]["product_output"]["frame_indices"]
        for crop in workload["quality"]["product_output"]["crops"]
    }
    for product_crops in crops.values():
        _require(
            len(product_crops) == len(crop_keys)
            and {(item["frame_index"], item["crop"]) for item in product_crops} == crop_keys,
            "Product-output visual crop set is incomplete",
        )
        for crop in product_crops:
            _verify_record(crop, root)
    _require(
        report.get("status") == ("invalid" if failures else "valid"),
        "Product gate status contradicts metrics",
    )
    _require(report.get("publishable") is (not failures), "Product gate publication status changed")
    _require(
        report.get("errors") == [f"TheAnimeScripter: {error}" for error in failures],
        "Product gate has unaccounted errors",
    )
    return failures


def _profile_result(
    directory: Path, *, root: Path, identity: dict[str, Any], candidate_id: str, returncode: int
) -> dict[str, Any]:
    _require(
        returncode in {0, 2}, f"Quality command failed with infrastructure exit code {returncode}"
    )
    profile = PROFILES[candidate_id]
    workload = load_manifest(_verify_record(identity["files"]["workload_manifest"], root))
    tensor_dir = directory / "tensor-quality"
    product_dir = directory / "product-output"
    # Even a saved inference failure must never hide a later invalid media/NVML run.
    for path in directory.glob("**/run-*/manifest.json"):
        run = load_json(path)
        _require(
            run.get("status") == "valid" and not run.get("errors"),
            f"Invalid quality run: {path}: {run.get('errors')}",
        )
        nvml = run.get("measured", {}).get("metrics", {}).get("nvml")
        if nvml is not None:
            _require(
                nvml.get("valid") is True and not nvml.get("errors"),
                f"Invalid quality NVML evidence: {path}: {nvml.get('errors')}",
            )
    _tensor_gate(tensor_dir, identity, profile, workload, preprocessing=True)
    failures = _tensor_gate(tensor_dir, identity, profile, workload, preprocessing=False)
    evidence_paths = {
        "preprocessing": tensor_dir / "preprocessing-diagnostic.json",
        "inference": tensor_dir / "inference-parity.json",
    }
    if not failures or (product_dir / "product-output-parity.json").exists():
        failures.extend(_product_gate(product_dir, root, identity, profile, workload))
        evidence_paths["product_output"] = product_dir / "product-output-parity.json"
    _require(
        returncode == (2 if failures else 0), "Quality command exit code contradicts its reports"
    )
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "execution_profile": profile,
        "identity": identity,
        "status": "disqualified" if failures else "valid",
        "errors": failures,
        "returncode": returncode,
        "evidence": {key: _record(path, root) for key, path in evidence_paths.items()},
        "artifacts": [
            _record(path, root)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "result.json"
        ],
    }


def _load_profile(
    path: Path, *, root: Path, identity: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    result = load_json(path)
    _require(result.get("identity") == identity, f"Stale preflight profile: {candidate_id}")
    for record in result["artifacts"]:
        _verify_record(record, root)
    expected = _profile_result(
        path.parent,
        root=root,
        identity=identity,
        candidate_id=candidate_id,
        returncode=result["returncode"],
    )
    _require(result == expected, f"Preflight profile evidence changed: {candidate_id}")
    return {**result, "result": _record(path, root)}


def _load_preflight_report(
    path: Path, *, root: Path, expected_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate hashed preflight evidence before using its eligible/disqualified profiles."""
    root = root.resolve()
    path = _path(root, path)
    report = load_json(path)
    _require(
        report.get("schema_version") == 1
        and report.get("document_type") == "tas-quality-preflight",
        "Unsupported TAS preflight report",
    )
    identity = report["identity"]
    if expected_identity is not None:
        _require(identity == expected_identity, "Stale TAS preflight identity")
    for record in identity["files"].values():
        _verify_record(record, root)
    _require(
        identity["repository_revision"] == _revision(root), "Preflight repository revision changed"
    )
    if expected_identity is None:
        for image in identity["images"].values():
            _require(_image_id(image["reference"]) == image["id"], "Preflight Docker image changed")
    profiles = report["profiles"]
    _require(set(profiles) == set(PROFILES), "TAS preflight profile set is incomplete")
    for candidate_id, profile in profiles.items():
        result_path = _verify_record(profile["result"], root)
        expected_path = path.parent / candidate_id / "result.json"
        _require(result_path == expected_path, "Preflight profile path changed")
        expected = _load_profile(
            result_path, root=root, identity=identity, candidate_id=candidate_id
        )
        _require(profile == expected, f"Preflight profile summary changed: {candidate_id}")
    eligible = [key for key, value in profiles.items() if value["status"] == "valid"]
    _require(report.get("eligible_candidates") == eligible, "Preflight eligibility changed")
    _require(
        report.get("status") == ("valid" if eligible else "invalid"), "Preflight status changed"
    )
    return report


def load_preflight_report(
    path: Path, *, root: Path, expected_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read the four profiles with mandatory source/assets/gate hash verification."""
    try:
        return _load_preflight_report(path, root=root, expected_identity=expected_identity)
    except PreflightError:
        raise
    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise PreflightError(f"Cannot validate TAS preflight evidence: {exc}") from exc


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run every I/O configuration; reuse only immutable, verified completed profiles."""
    root = Path(args.root).resolve()
    manifest = _path(root, args.manifest)
    identity = build_preflight_identity(
        root=root,
        manifest=manifest,
        variant=args.variant,
        engine=Path(args.engine),
        tas_engine=Path(args.tas_engine),
        gpu_id=args.gpu_id,
    )
    output = _path(
        root,
        args.output_dir
        or (
            f"artefacts/benchmarks/comparative/tuning/{manifest.stem}-{args.variant}/tas-preflight"
        ),
    )
    _require(
        root / "artefacts" in output.parents,
        "Preflight output must be under artefacts for Docker mounts",
    )
    report_path = output / "preflight.json"
    if report_path.exists():
        return load_preflight_report(report_path, root=root, expected_identity=identity)
    config_path = output / "preflight.config.json"
    config = {"schema_version": 1, "identity": identity, "profiles": PROFILES}
    if config_path.exists():
        _require(
            load_json(config_path) == config,
            "Stale preflight configuration; use another output directory",
        )
    else:
        _require(
            not output.exists() or not any(output.iterdir()),
            "Preflight output contains partial evidence without a config",
        )
        _immutable_json(config_path, config)
    profiles = {}
    for index, (candidate_id, profile) in enumerate(PROFILES.items(), 1):
        directory = output / candidate_id
        result_path = directory / "result.json"
        print(f"[TAS preflight {index}/4] {candidate_id}", flush=True)
        if not result_path.exists():
            _require(
                not directory.exists() or not any(directory.iterdir()),
                f"Partial preflight evidence must be archived before retrying: {directory}",
            )
            variables = {
                "MANIFEST": manifest.relative_to(root).as_posix(),
                "VARIANT": args.variant,
                "ENGINE": identity["files"]["engine"]["path"],
                "TAS_ENGINE": identity["files"]["tas_engine"]["path"],
                "GPU_ID": str(args.gpu_id),
                "EXECUTION_PROFILE": "tuned",
                "VSTRT_ARGS": VSTRT_ARGS,
                "TAS_ARGS": (
                    f"--decode-method {profile['decode_method']} --writer {profile['writer']}"
                ),
                "MODEL_SPACE_DIR": (directory / "tensor-quality").relative_to(root).as_posix(),
                "PRODUCT_OUTPUT_DIR": (directory / "product-output").relative_to(root).as_posix(),
                "BENCHMARK_IMAGE": identity["images"]["trtvideo"]["reference"],
                "VSTRT_IMAGE": identity["images"]["vstrt"]["reference"],
                "TAS_IMAGE": identity["images"]["tas"]["reference"],
            }
            result = subprocess.run(
                [
                    "make",
                    "-C",
                    str(root / "benchmarks"),
                    "quality-gates",
                    "-j1",
                    *[f"{key}={value}" for key, value in variables.items()],
                ],
                cwd=root,
                check=False,
            )
            profile_result = _profile_result(
                directory,
                root=root,
                identity=identity,
                candidate_id=candidate_id,
                returncode=result.returncode,
            )
            _immutable_json(result_path, profile_result)
        else:
            print("  SKIP verified quality evidence", flush=True)
        profiles[candidate_id] = _load_profile(
            result_path, root=root, identity=identity, candidate_id=candidate_id
        )
        print(f"  {profiles[candidate_id]['status']}", flush=True)
    eligible = [key for key, value in profiles.items() if value["status"] == "valid"]
    report = {
        "schema_version": 1,
        "document_type": "tas-quality-preflight",
        "status": "valid" if eligible else "invalid",
        "identity": identity,
        "profiles": profiles,
        "eligible_candidates": eligible,
    }
    _immutable_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["720p", "1080p"], required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tas-engine", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    try:
        report = run_preflight(build_parser().parse_args())
    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: TAS preflight infrastructure/evidence failure: {exc}", file=sys.stderr)
        sys.exit(2)
    print(
        f"TAS quality preflight {report['status']}: "
        f"{len(report['eligible_candidates'])}/4 eligible",
        flush=True,
    )
    if not report["eligible_candidates"]:
        print("ERROR: No eligible TAS I/O configuration", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
