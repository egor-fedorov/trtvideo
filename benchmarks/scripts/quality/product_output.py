"""Final MP4 comparison and visual crop generation."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_media.benchmarking.environment import relative_artifact_path, sha256_file

REPORT_SCHEMA_VERSION = 1
_PSNR_PATTERN = re.compile(r"\baverage:(inf|[0-9]+(?:\.[0-9]+)?)")
_SSIM_PATTERN = re.compile(r"\bAll:([0-9]+(?:\.[0-9]+)?)")


class ProductOutputError(RuntimeError):
    """Raised when retained MP4 evidence cannot form a quality comparison."""


@dataclass(frozen=True)
class ProductOutputThresholds:
    """Fixed acceptance thresholds for decoded product outputs."""

    psnr_min_db: float
    ssim_min: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProductOutputThresholds:
        try:
            thresholds = cls(
                psnr_min_db=float(value["psnr_min_db"]),
                ssim_min=float(value["ssim_min"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductOutputError(f"Invalid product-output thresholds: {exc}") from exc
        if thresholds.psnr_min_db <= 0 or not 0 < thresholds.ssim_min <= 1:
            raise ProductOutputError("Invalid product-output threshold range")
        return thresholds

    def as_dict(self) -> dict[str, float]:
        return {
            "psnr_min_db": self.psnr_min_db,
            "ssim_min": self.ssim_min,
        }


@dataclass(frozen=True)
class OutputEvidence:
    """One retained, validated output and its inference contract."""

    product: str
    workload_id: str
    variant: str
    frames: int
    encoder: dict[str, Any]
    input_sha256: str
    onnx_sha256: str
    engine_sha256: str
    output_path: Path
    output_sha256: str
    manifest_path: Path

    @classmethod
    def load(cls, manifest_path: Path, *, root: Path) -> OutputEvidence:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductOutputError(
                f"Cannot read output manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ProductOutputError(f"Output manifest must be an object: {manifest_path}")
        if manifest.get("status") != "valid":
            raise ProductOutputError(f"Output run is not valid: {manifest_path}")
        if manifest.get("reproducibility", {}).get("publishable") is not True:
            raise ProductOutputError(
                f"Output run is not reproducible: {manifest_path}"
            )
        validation = manifest.get("measured", {}).get("validation", {})
        if validation.get("valid") is not True:
            raise ProductOutputError(
                f"Output media validation failed: {manifest_path}"
            )
        output = manifest.get("measured", {}).get("output")
        if not isinstance(output, dict):
            raise ProductOutputError(
                f"Output run did not retain its measured MP4: {manifest_path}"
            )
        relative_path = output.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ProductOutputError(f"Output manifest has no MP4 path: {manifest_path}")
        output_path = (root / relative_path).resolve()
        if output_path != root and root not in output_path.parents:
            raise ProductOutputError("Retained MP4 path escapes the artifact root")
        if not output_path.is_file():
            raise ProductOutputError(f"Retained MP4 not found: {output_path}")
        output_sha256 = output.get("sha256")
        if output_sha256 != sha256_file(output_path):
            raise ProductOutputError(f"Retained MP4 SHA256 changed: {output_path}")

        assets = manifest.get("assets", {})
        parameters = manifest.get("parameters", {})
        try:
            evidence = cls(
                product=str(manifest["product"]),
                workload_id=str(manifest["workload_id"]),
                variant=str(manifest["variant"]),
                frames=int(parameters["frames"]),
                encoder=dict(parameters["encoder"]),
                input_sha256=str(assets["input"]["sha256"]),
                onnx_sha256=str(assets["onnx"]["sha256"]),
                engine_sha256=str(assets["engine"]["sha256"]),
                output_path=output_path,
                output_sha256=str(output_sha256),
                manifest_path=manifest_path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductOutputError(
                f"Output manifest contract is incomplete: {manifest_path}: {exc}"
            ) from exc
        return evidence


def validate_evidence_set(
    reference: OutputEvidence,
    candidates: list[OutputEvidence],
    *,
    expected_frames: int,
) -> None:
    """Require identical source/model/encoder contracts for all outputs."""
    if reference.product != "ai-media-enhancer":
        raise ProductOutputError("Product-output reference must be ai-media-enhancer")
    if reference.frames != expected_frames:
        raise ProductOutputError("Reference output does not contain canonical frames")
    expected_products = {"vs-mlrt", "VSGAN-tensorrt-docker"}
    actual_products = {candidate.product for candidate in candidates}
    if actual_products != expected_products or len(candidates) != len(expected_products):
        raise ProductOutputError(
            "Product-output candidates must be exactly vs-mlrt and "
            "VSGAN-tensorrt-docker"
        )
    for candidate in candidates:
        checks = {
            "workload": (candidate.workload_id, reference.workload_id),
            "variant": (candidate.variant, reference.variant),
            "frame count": (candidate.frames, expected_frames),
            "encoder": (candidate.encoder, reference.encoder),
            "input SHA256": (candidate.input_sha256, reference.input_sha256),
            "ONNX SHA256": (candidate.onnx_sha256, reference.onnx_sha256),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise ProductOutputError(f"{candidate.product} changed {label}")
        if candidate.product == "vs-mlrt" and (
            candidate.engine_sha256 != reference.engine_sha256
        ):
            raise ProductOutputError("vs-mlrt changed the parity engine")


def build_metric_command(
    reference: Path,
    candidate: Path,
    *,
    metric: str,
    stats_path: Path,
) -> list[str]:
    """Build one complete two-input FFmpeg quality decode."""
    if metric not in {"psnr", "ssim"}:
        raise ProductOutputError(f"Unknown product-output metric: {metric}")
    filter_graph = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS[reference];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS[candidate];"
        f"[reference][candidate]{metric}=stats_file={stats_path}"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(reference),
        "-i",
        str(candidate),
        "-filter_complex",
        filter_graph,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]


def parse_metric_log(metric: str, value: str) -> dict[str, Any]:
    """Parse the final FFmpeg PSNR or SSIM summary."""
    pattern = _PSNR_PATTERN if metric == "psnr" else _SSIM_PATTERN
    matches = pattern.findall(value)
    if not matches:
        raise ProductOutputError(f"FFmpeg {metric} summary was not found")
    result = matches[-1]
    if metric == "psnr":
        exact = result == "inf"
        return {
            "exact": exact,
            "average_db": None if exact else float(result),
        }
    return {"all": float(result)}


def run_metric(
    reference: Path,
    candidate: Path,
    *,
    metric: str,
    output_dir: Path,
    expected_frames: int,
    root: Path,
) -> dict[str, Any]:
    """Run one complete decoded MP4 metric and retain its raw statistics."""
    stats_path = output_dir / f"{metric}.stats.log"
    log_path = output_dir / f"{metric}.ffmpeg.log"
    command = build_metric_command(
        reference,
        candidate,
        metric=metric,
        stats_path=stats_path,
    )
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=log,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise ProductOutputError(f"Cannot start FFmpeg {metric}: {exc}") from exc
    if result.returncode != 0:
        raise ProductOutputError(f"FFmpeg {metric} failed; see {log_path}")
    if not stats_path.is_file():
        raise ProductOutputError(f"FFmpeg {metric} did not create {stats_path}")
    frame_count = sum(
        1
        for line in stats_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if frame_count != expected_frames:
        raise ProductOutputError(
            f"FFmpeg {metric} compared {frame_count} frames, "
            f"expected {expected_frames}"
        )
    metrics = parse_metric_log(metric, log_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "frames": frame_count,
            "stats_path": relative_artifact_path(stats_path, root),
            "stats_sha256": sha256_file(stats_path),
            "ffmpeg_log": relative_artifact_path(log_path, root),
            "ffmpeg_log_sha256": sha256_file(log_path),
        }
    )
    return metrics


def crop_geometry(
    crop: dict[str, Any],
    *,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    """Resolve a normalized crop to an in-frame integer rectangle."""
    width = max(1, round(output_width * float(crop["width"])))
    height = max(1, round(output_height * float(crop["height"])))
    x = round(output_width * float(crop["x"]))
    y = round(output_height * float(crop["y"]))
    x = min(x, output_width - width)
    y = min(y, output_height - height)
    return x, y, width, height


def build_crop_command(
    input_path: Path,
    outputs: list[tuple[Path, int, tuple[int, int, int, int]]],
) -> list[str]:
    """Build one FFmpeg decode that emits every requested visual crop."""
    if not outputs:
        raise ProductOutputError("At least one visual crop is required")
    split_outputs = "".join(f"[split{index}]" for index in range(len(outputs)))
    filters = [f"[0:v]split={len(outputs)}{split_outputs}"]
    for index, (_, frame_index, geometry) in enumerate(outputs):
        x, y, width, height = geometry
        filters.append(
            f"[split{index}]select=eq(n\\,{frame_index}),"
            f"crop={width}:{height}:{x}:{y},setpts=PTS-STARTPTS[crop{index}]"
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        ";".join(filters),
    ]
    for index, (path, _, _) in enumerate(outputs):
        command.extend(
            [
                "-map",
                f"[crop{index}]",
                "-frames:v",
                "1",
                "-c:v",
                "png",
                "-update",
                "1",
                str(path),
            ]
        )
    return command


def generate_visual_crops(
    evidence: OutputEvidence,
    *,
    output_dir: Path,
    frame_indices: list[int],
    crops: list[dict[str, Any]],
    output_width: int,
    output_height: int,
    root: Path,
) -> list[dict[str, Any]]:
    """Decode once and write the canonical crop matrix for one product."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metadata = []
    for frame_index in frame_indices:
        for crop in crops:
            geometry = crop_geometry(
                crop,
                output_width=output_width,
                output_height=output_height,
            )
            path = output_dir / f"frame-{frame_index:06d}.{crop['name']}.png"
            outputs.append((path, frame_index, geometry))
            metadata.append((path, frame_index, crop["name"], geometry))
    command = build_crop_command(evidence.output_path, outputs)
    log_path = output_dir / "crops.ffmpeg.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=log,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise ProductOutputError(f"Cannot start FFmpeg crop extraction: {exc}") from exc
    if result.returncode != 0:
        raise ProductOutputError(f"FFmpeg crop extraction failed; see {log_path}")

    artifacts = []
    for path, frame_index, crop_name, geometry in metadata:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ProductOutputError(f"Visual crop was not created: {path}")
        artifacts.append(
            {
                "frame_index": frame_index,
                "crop": crop_name,
                "geometry": {
                    "x": geometry[0],
                    "y": geometry[1],
                    "width": geometry[2],
                    "height": geometry[3],
                },
                "path": relative_artifact_path(path, root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


def compare_product_outputs(
    reference_manifest: Path,
    candidate_manifests: list[Path],
    *,
    workload: dict[str, Any],
    variant: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare retained final MP4 outputs and build a quality-gate report."""
    reference = OutputEvidence.load(reference_manifest, root=root)
    candidates = [
        OutputEvidence.load(manifest_path, root=root)
        for manifest_path in candidate_manifests
    ]
    quality = workload["quality"]["product_output"]
    frame_indices = list(quality["frame_indices"])
    thresholds = ProductOutputThresholds.from_dict(quality["thresholds"])
    validate_evidence_set(
        reference,
        candidates,
        expected_frames=int(workload["clip"]["frames"]),
    )
    if reference.workload_id != workload["id"] or reference.variant != variant["name"]:
        raise ProductOutputError(
            "Retained outputs do not match the canonical workload variant"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_artifacts = {
        reference.product: generate_visual_crops(
            reference,
            output_dir=output_dir / "crops" / "ai-media",
            frame_indices=frame_indices,
            crops=quality["crops"],
            output_width=int(variant["benchmark_output"]["width"]),
            output_height=int(variant["benchmark_output"]["height"]),
            root=root,
        )
    }
    comparisons = []
    report_errors: list[str] = []
    for candidate in candidates:
        candidate_dir = output_dir / (
            "vstrt" if candidate.product == "vs-mlrt" else "vsgan"
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)
        psnr = run_metric(
            reference.output_path,
            candidate.output_path,
            metric="psnr",
            output_dir=candidate_dir,
            expected_frames=reference.frames,
            root=root,
        )
        ssim = run_metric(
            reference.output_path,
            candidate.output_path,
            metric="ssim",
            output_dir=candidate_dir,
            expected_frames=reference.frames,
            root=root,
        )
        errors = []
        psnr_value = math.inf if psnr["exact"] else float(psnr["average_db"])
        if psnr_value < thresholds.psnr_min_db:
            errors.append(
                f"PSNR must be >= {thresholds.psnr_min_db:g} dB, "
                f"got {psnr_value:g} dB"
            )
        if float(ssim["all"]) < thresholds.ssim_min:
            errors.append(
                f"SSIM must be >= {thresholds.ssim_min:g}, got {ssim['all']:g}"
            )
        crop_key = "vstrt" if candidate.product == "vs-mlrt" else "vsgan"
        crop_artifacts[candidate.product] = generate_visual_crops(
            candidate,
            output_dir=output_dir / "crops" / crop_key,
            frame_indices=frame_indices,
            crops=quality["crops"],
            output_width=int(variant["benchmark_output"]["width"]),
            output_height=int(variant["benchmark_output"]["height"]),
            root=root,
        )
        comparisons.append(
            {
                "implementation": candidate.product,
                "engine_sha256": candidate.engine_sha256,
                "output_sha256": candidate.output_sha256,
                "run_manifest": relative_artifact_path(
                    candidate.manifest_path,
                    root,
                ),
                "run_manifest_sha256": sha256_file(candidate.manifest_path),
                "metrics": {
                    "psnr": psnr,
                    "ssim": ssim,
                },
                "status": "valid" if not errors else "invalid",
                "errors": errors,
            }
        )
        report_errors.extend(f"{candidate.product}: {error}" for error in errors)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "document_type": "product-output-parity",
        "status": "valid" if not report_errors else "invalid",
        "publishable": not report_errors,
        "workload_id": reference.workload_id,
        "variant": reference.variant,
        "frame_indices": frame_indices,
        "thresholds": thresholds.as_dict(),
        "assets": {
            "input_sha256": reference.input_sha256,
            "onnx_sha256": reference.onnx_sha256,
        },
        "reference": {
            "implementation": reference.product,
            "engine_sha256": reference.engine_sha256,
            "output_sha256": reference.output_sha256,
            "run_manifest": relative_artifact_path(reference.manifest_path, root),
            "run_manifest_sha256": sha256_file(reference.manifest_path),
        },
        "comparisons": comparisons,
        "visual_crops": crop_artifacts,
        "errors": report_errors,
    }
