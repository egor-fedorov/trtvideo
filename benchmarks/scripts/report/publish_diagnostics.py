"""Build a compact diagnostic publication from raw trtexec and Nsight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any

CANONICAL_ROOT = PurePosixPath("artefacts/benchmarks/diagnostics")
DEFAULT_OUTPUT = Path("benchmarks/results/rtx-3090/diagnostics.json")
TRTEXEC_WORKLOADS = (
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "realesrgan-x2plus-madrid-v1", "720p"),
    ("realesrgan_x2plus_madrid", "RealESRGAN_x2plus", "realesrgan-x2plus-madrid-v1", "1080p"),
    ("liveaction_span_madrid", "SPAN", "liveaction-span-madrid-v1", "720p"),
    ("liveaction_span_madrid", "SPAN", "liveaction-span-madrid-v1", "1080p"),
)
NSIGHT_WORKLOAD = ("liveaction_span_madrid", "liveaction-span-madrid-v1", "1080p")
REQUIRED_SQLITE_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "ENUM_CUDA_MEMCPY_OPER",
    "ENUM_VIDEO_ENGINE_TYPE",
    "GPU_VIDEO_ENGINE_WORKLOAD",
    "NVTX_EVENTS",
    "StringIds",
}

Interval = tuple[int, int]


class DiagnosticPublicationError(RuntimeError):
    """Diagnostic evidence is incomplete, inconsistent, or invalid."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticPublicationError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticPublicationError(f"Expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DiagnosticPublicationError(f"Cannot hash {path}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticPublicationError(message)


def _source_is_clean(environment: dict[str, Any]) -> bool:
    return environment["image"].get("source_dirty") in (False, "0")


class EvidenceSource:
    """Resolve copied diagnostics while retaining canonical artifact paths."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def canonical(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise DiagnosticPublicationError(f"Evidence escapes diagnostics root: {path}") from exc
        return str(CANONICAL_ROOT / PurePosixPath(relative.as_posix()))

    def resolve(self, path: str) -> Path:
        try:
            relative = PurePosixPath(path).relative_to(CANONICAL_ROOT)
        except ValueError as exc:
            raise DiagnosticPublicationError(f"Artifact path escapes diagnostics: {path}") from exc
        return self.root / Path(*relative.parts)


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _interval_duration(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _overlap_duration(left: list[Interval], right: list[Interval]) -> int:
    left_merged = _merge_intervals(left)
    right_merged = _merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _clipped_intervals(
    connection: sqlite3.Connection,
    table: str,
    start: int,
    end: int,
    *,
    clause: str = "",
    parameters: tuple[Any, ...] = (),
) -> list[Interval]:
    query = f"SELECT MAX(start, ?), MIN(end, ?) FROM {table} WHERE end > ? AND start < ?"
    if clause:
        query += f" AND {clause}"
    rows = connection.execute(query, (start, end, start, end, *parameters))
    return [(int(row[0]), int(row[1])) for row in rows if row[1] > row[0]]


def _enum_id(connection: sqlite3.Connection, table: str, name: str) -> int:
    rows = connection.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchall()
    if len(rows) != 1:
        raise DiagnosticPublicationError(f"Nsight SQLite lacks one {table}.{name} value")
    return int(rows[0][0])


def _frame_loop(connection: sqlite3.Connection) -> Interval:
    rows = connection.execute(
        """
        SELECT event.start, event.end
        FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS string ON string.id = event.textId
        WHERE COALESCE(string.value, event.text) = 'trtvideo.frame_loop'
        """
    ).fetchall()
    if len(rows) != 1 or rows[0][1] is None or rows[0][1] <= rows[0][0]:
        raise DiagnosticPublicationError("Nsight SQLite lacks one complete trtvideo.frame_loop")
    return int(rows[0][0]), int(rows[0][1])


def _copy_rows(
    connection: sqlite3.Connection,
    copy_kind: int,
    start: int,
    end: int,
) -> list[tuple[int, int, int]]:
    rows = connection.execute(
        """
        SELECT MAX(start, ?), MIN(end, ?), bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = ? AND end > ? AND start < ?
        """,
        (start, end, copy_kind, start, end),
    )
    return [(int(row[0]), int(row[1]), int(row[2])) for row in rows if row[1] > row[0]]


def analyze_nsight_sqlite(path: Path, *, frames: int) -> dict[str, Any]:
    """Derive frame-loop overlap and copy findings from an Nsight SQLite export."""
    _require(frames > 0, "Nsight frame count must be positive")
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise DiagnosticPublicationError(f"Cannot open Nsight SQLite {path}: {exc}") from exc
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = sorted(REQUIRED_SQLITE_TABLES - tables)
        _require(not missing, f"Nsight SQLite lacks tables: {', '.join(missing)}")

        frame_start, frame_end = _frame_loop(connection)
        frame_duration = frame_end - frame_start
        kernels = _clipped_intervals(
            connection,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            frame_start,
            frame_end,
        )
        _require(bool(kernels), "Nsight frame loop contains no CUDA kernels")
        kernel_duration = _interval_duration(kernels)

        video_overlap: dict[str, float] = {}
        for label, enum_name in (("nvdec", "NVDEC"), ("nvenc", "NVENC")):
            engine_id = _enum_id(connection, "ENUM_VIDEO_ENGINE_TYPE", enum_name)
            intervals = _clipped_intervals(
                connection,
                "GPU_VIDEO_ENGINE_WORKLOAD",
                frame_start,
                frame_end,
                clause="engineType = ?",
                parameters=(engine_id,),
            )
            duration = _interval_duration(intervals)
            _require(duration > 0, f"Nsight frame loop contains no {enum_name} workload")
            video_overlap[label] = _overlap_duration(intervals, kernels) / duration * 100.0

        copy_kinds = {
            label: _enum_id(connection, "ENUM_CUDA_MEMCPY_OPER", enum_name)
            for label, enum_name in (
                ("h2d", "CUDA_MEMCPY_KIND_HTOD"),
                ("d2h", "CUDA_MEMCPY_KIND_DTOH"),
                ("d2d", "CUDA_MEMCPY_KIND_DTOD"),
            )
        }
        copies = {
            label: _copy_rows(connection, kind, frame_start, frame_end)
            for label, kind in copy_kinds.items()
        }
        startup_h2d = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(bytes), 0)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind = ? AND end <= ?
            """,
            (copy_kinds["h2d"], frame_start),
        ).fetchone()
        _require(startup_h2d is not None, "Cannot summarize startup H2D copies")

        h2d_bytes = sum(row[2] for row in copies["h2d"])
        d2h_bytes = sum(row[2] for row in copies["d2h"])
        d2d_bytes = sum(row[2] for row in copies["d2d"])
        d2d_duration = sum(row[1] - row[0] for row in copies["d2d"])
        return {
            "frame_loop_sec": frame_duration / 1_000_000_000.0,
            "cuda_kernel_time_coverage_of_frame_loop_percent": (
                kernel_duration / frame_duration * 100.0
            ),
            "nvdec_overlap_with_cuda_kernels_percent": video_overlap["nvdec"],
            "nvenc_overlap_with_cuda_kernels_percent": video_overlap["nvenc"],
            "startup_h2d_copy_count": int(startup_h2d[0]),
            "startup_h2d_mib": int(startup_h2d[1]) / 1024.0**2,
            "frame_loop_h2d_copy_count": len(copies["h2d"]),
            "frame_loop_h2d_mib": h2d_bytes / 1024.0**2,
            "frame_loop_d2h_copy_count": len(copies["d2h"]),
            "frame_loop_d2h_mib": d2h_bytes / 1024.0**2,
            "frame_loop_d2d_copy_count": len(copies["d2d"]),
            "per_frame_d2d_mib": d2d_bytes / frames / 1024.0**2,
            "per_frame_d2d_time_ms": d2d_duration / frames / 1_000_000.0,
            "material_per_frame_h2d_or_d2h": bool(h2d_bytes or d2h_bytes),
        }
    except sqlite3.Error as exc:
        raise DiagnosticPublicationError(f"Cannot analyze Nsight SQLite {path}: {exc}") from exc
    finally:
        connection.close()


def _environment_contract(environment: dict[str, Any]) -> dict[str, Any]:
    return {key: environment[key] for key in ("cpu", "gpu", "image")}


def _compact_trtexec(
    source: EvidenceSource,
    base: str,
    workload_name: str,
    workload_id: str,
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite_path = source.root / "trtexec" / f"{base}-{variant}" / "suite.json"
    suite = _load(suite_path)
    _require(suite.get("document_type") == "benchmark-result", f"Invalid suite: {suite_path}")
    _require(suite.get("product") == "trtexec", f"Unexpected product: {suite_path}")
    _require(suite.get("status") == "valid", f"Invalid trtexec suite: {suite_path}")
    _require(suite.get("publishable") is True, f"Unpublishable trtexec suite: {suite_path}")
    _require(suite.get("errors") == [], f"trtexec suite has errors: {suite_path}")
    _require(suite.get("workload_id") == workload_id, f"Wrong workload: {suite_path}")
    _require(suite.get("variant") == variant, f"Wrong variant: {suite_path}")
    _require(suite.get("benchmark_contract_version") == 2, f"Wrong contract: {suite_path}")
    parameters = suite.get("parameters", {})
    _require(parameters.get("frames") == 1000, f"Wrong trtexec frame count: {suite_path}")
    _require(parameters.get("cuda_graph") is False, f"CUDA Graph enabled: {suite_path}")
    _require(parameters.get("data_transfers") is False, f"Transfers enabled: {suite_path}")

    manifests = []
    manifest_paths = []
    for run in suite.get("runs", []):
        _require(run.get("status") == "valid", f"Invalid trtexec run in {suite_path}")
        manifest_path = source.resolve(str(run["manifest"]))
        manifest = _load(manifest_path)
        _require(manifest.get("status") == "valid", f"Invalid run manifest: {manifest_path}")
        _require(manifest.get("errors") == [], f"Run manifest has errors: {manifest_path}")
        _require(
            manifest.get("reproducibility", {}).get("publishable") is True,
            f"Run is not reproducible: {manifest_path}",
        )
        _require(manifest.get("workload_id") == workload_id, f"Wrong run workload: {manifest_path}")
        _require(manifest.get("variant") == variant, f"Wrong run variant: {manifest_path}")
        _require(manifest.get("parameters") == parameters, f"Run parameters drift: {manifest_path}")
        _require(_source_is_clean(manifest["environment"]), f"Dirty run source: {manifest_path}")
        manifests.append(manifest)
        manifest_paths.append(manifest_path)
    _require(bool(manifests), f"trtexec suite has no runs: {suite_path}")

    environment = manifests[0]["environment"]
    assets = manifests[0]["assets"]
    for manifest, manifest_path in zip(manifests[1:], manifest_paths[1:], strict=True):
        _require(manifest["environment"] == environment, f"Run environment drift: {manifest_path}")
        _require(manifest["assets"] == assets, f"Run asset drift: {manifest_path}")

    nvml = [manifest["metrics"]["nvml"] for manifest in manifests]
    result = {
        "workload_id": workload_id,
        "workload": workload_name,
        "variant": variant,
        "benchmark_contract_version": 2,
        "status": "valid",
        "publishable": True,
        "suite": {"path": source.canonical(suite_path), "sha256": _digest(suite_path)},
        "run_manifest_sha256": [_digest(path) for path in manifest_paths],
        "assets": assets,
        "parameters": parameters,
        "statistics": suite["statistics"],
        "session_observations": {
            "peak_temperature_c": max(item["temperature"]["peak_c"] for item in nvml),
            "power_cap_observed": any(item["power"]["power_cap_observed"] for item in nvml),
            "throttle_reasons": sorted(
                {reason for item in nvml for reason in item["throttle_reasons"]}
            ),
        },
    }
    return result, environment


def _compact_nsight(
    source: EvidenceSource,
    expected_environment: dict[str, Any],
    revision: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    base, workload_id, variant = NSIGHT_WORKLOAD
    directory = source.root / "nsight" / f"{base}-{variant}"
    manifest_path = directory / "manifest.json"
    manifest = _load(manifest_path)
    _require(manifest.get("document_type") == "nsight-diagnostic", "Wrong Nsight document")
    _require(manifest.get("scope") == "diagnostic", "Wrong Nsight scope")
    _require(manifest.get("status") == "valid", "Nsight diagnostic is invalid")
    _require(manifest.get("publishable") is False, "Nsight trace cannot be a performance result")
    _require(manifest.get("errors") == [], "Nsight diagnostic has errors")
    _require(manifest.get("workload_id") == workload_id, "Wrong Nsight workload")
    _require(manifest.get("variant") == variant, "Wrong Nsight variant")
    _require(manifest.get("benchmark_contract_version") == 2, "Wrong Nsight contract")
    _require(manifest.get("profile", {}).get("returncode") == 0, "Nsight command failed")
    validation = manifest.get("profile", {}).get("output_validation", {})
    _require(validation.get("valid") is True, "Nsight output validation failed")
    _require(all(validation.get("checks", {}).values()), "Nsight output checks are incomplete")

    environment = manifest["environment"]
    _require(_source_is_clean(environment), "Nsight source is dirty")
    _require(environment["image"]["repository_revision"] == revision, "Nsight revision drift")
    _require(
        _environment_contract(environment) == _environment_contract(expected_environment),
        "Nsight hardware or image differs from trtexec",
    )
    for key, value in expected_environment["software"].items():
        _require(environment["software"].get(key) == value, f"Nsight software drift: {key}")

    trace_path = source.resolve(str(manifest["artifacts"]["trace"]))
    sqlite_path = source.resolve(str(manifest["artifacts"]["sqlite"]))
    _require(trace_path.is_file() and trace_path.stat().st_size > 0, "Nsight trace is missing")
    _require(sqlite_path.is_file() and sqlite_path.stat().st_size > 0, "Nsight SQLite is missing")
    frames = int(manifest["parameters"]["frames"])
    findings = analyze_nsight_sqlite(sqlite_path, frames=frames)

    result = {
        "status": "valid",
        "publishable_as_performance_result": False,
        "workload_id": workload_id,
        "variant": variant,
        "benchmark_contract_version": 2,
        "frames": frames,
        "evidence": {
            "manifest": {
                "path": source.canonical(manifest_path),
                "sha256": _digest(manifest_path),
            },
            "trace": {
                "path": source.canonical(trace_path),
                "sha256": _digest(trace_path),
                "size_bytes": trace_path.stat().st_size,
            },
            "sqlite": {
                "path": source.canonical(sqlite_path),
                "sha256": _digest(sqlite_path),
                "size_bytes": sqlite_path.stat().st_size,
            },
        },
        "output_validation": True,
        "findings": findings,
        "limitations": manifest["limitations"],
    }
    return result, str(manifest["started_at_utc"])[:10], environment


def build_document(source: EvidenceSource) -> dict[str, Any]:
    trtexec = []
    environments = []
    for base, workload_name, workload_id, variant in TRTEXEC_WORKLOADS:
        result, environment = _compact_trtexec(
            source,
            base,
            workload_name,
            workload_id,
            variant,
        )
        trtexec.append(result)
        environments.append(environment)

    reference_environment = environments[0]
    for environment in environments[1:]:
        _require(environment == reference_environment, "trtexec environment drift across workloads")
    revision = str(reference_environment["image"]["repository_revision"])
    _require(len(revision) == 40, "Invalid diagnostic repository revision")
    nsight, date_utc, nsight_environment = _compact_nsight(
        source,
        reference_environment,
        revision,
    )

    return {
        "schema_version": 3,
        "document_type": "published_diagnostic_results",
        "status": "valid",
        "publishable": True,
        "scope": {
            "date_utc": date_utc,
            "measurement_revision": revision,
            "source_dirty": False,
            "claim_scope": (
                "TensorRT inference ceilings and one profiler trace; diagnostics are not "
                "product competitors or additional throughput campaigns."
            ),
        },
        "environment": {
            "cpu": nsight_environment["cpu"],
            "gpu": nsight_environment["gpu"],
            "image": nsight_environment["image"],
            "software": nsight_environment["software"],
        },
        "methodology": {
            "trtexec_scope": "Inference-only ceiling with CUDA Graph and transfers disabled",
            "engine_scope": (
                "The diagnostics workflow builds fresh equivalent-contract TensorRT engines. "
                "Their hashes identify this diagnostic class and are not assumed to match "
                "engines from a separate tuned workflow."
            ),
            "cross_class_pipeline_efficiency_derived": False,
        },
        "trtexec": trtexec,
        "nsight": nsight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document = build_document(EvidenceSource(args.source_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Published diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
