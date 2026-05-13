"""Model/engine registry lookup for video inference."""

import json
import os
from dataclasses import dataclass
from typing import Any

from upscaler.video.info import VideoInfo

REGISTRY_FILENAME = "manifest.json"


@dataclass(frozen=True)
class EngineRegistryEntry:
    """Resolved engine registry entry."""

    engine_path: str
    manifest_path: str | None
    precision: str | None
    io_precision: str | None
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    tensorrt_version: str | None = None
    engine_sha256: str | None = None
    model_sha256: str | None = None

    @property
    def input_resolution(self) -> tuple[int, int] | None:
        if len(self.input_shape) != 4:
            return None
        _, _, height, width = self.input_shape
        if height <= 0 or width <= 0:
            return None
        return width, height

    @property
    def output_resolution(self) -> tuple[int, int] | None:
        if len(self.output_shape) != 4:
            return None
        _, _, height, width = self.output_shape
        if height <= 0 or width <= 0:
            return None
        return width, height


def default_registry_path(model_path: str) -> str:
    """Return registry manifest path from a model directory or direct JSON path."""
    if os.path.isdir(model_path):
        return os.path.join(model_path, REGISTRY_FILENAME)
    return model_path


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Registry JSON must be an object: {path}")
    return data


def _resolve_path(base_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    registry_relative = os.path.normpath(os.path.join(base_dir, path))
    if os.path.exists(registry_relative):
        return registry_relative
    return os.path.normpath(path)


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        return ()
    shape: list[int] = []
    for dim in value:
        if not isinstance(dim, int):
            return ()
        shape.append(dim)
    return tuple(shape)


def _entry_from_manifest(data: dict[str, Any], base_dir: str) -> EngineRegistryEntry | None:
    engine_path = data.get("engine_path")
    if not isinstance(engine_path, str):
        return None

    input_info = data.get("input")
    output_info = data.get("output")
    if not isinstance(input_info, dict) or not isinstance(output_info, dict):
        return None

    resolved_engine = _resolve_path(base_dir, engine_path)
    if resolved_engine is None:
        return None

    manifest_path = data.get("manifest_path")
    precision = data.get("precision")
    io_precision = data.get("io_precision")
    tensorrt_version = data.get("tensorrt_version")
    engine_sha256 = data.get("engine_sha256")
    model_sha256 = data.get("model_sha256")

    return EngineRegistryEntry(
        engine_path=resolved_engine,
        manifest_path=_resolve_path(base_dir, manifest_path)
        if isinstance(manifest_path, str)
        else None,
        precision=precision if isinstance(precision, str) else None,
        io_precision=io_precision if isinstance(io_precision, str) else None,
        input_shape=_shape(input_info.get("shape")),
        output_shape=_shape(output_info.get("shape")),
        tensorrt_version=tensorrt_version if isinstance(tensorrt_version, str) else None,
        engine_sha256=engine_sha256 if isinstance(engine_sha256, str) else None,
        model_sha256=model_sha256 if isinstance(model_sha256, str) else None,
    )


def _entry_from_registry_item(item: Any, registry_dir: str) -> EngineRegistryEntry | None:
    if not isinstance(item, dict):
        return None

    manifest_path = item.get("manifest_path")
    if isinstance(manifest_path, str):
        resolved_manifest = _resolve_path(registry_dir, manifest_path)
        if resolved_manifest and os.path.exists(resolved_manifest):
            manifest = _read_json(resolved_manifest)
            manifest.setdefault("manifest_path", manifest_path)
            return _entry_from_manifest(manifest, os.path.dirname(resolved_manifest))

    return _entry_from_manifest(item, registry_dir)


def load_engine_registry(model_path: str) -> list[EngineRegistryEntry]:
    """Load engine entries from a model directory, registry JSON, or sidecar manifest."""
    registry_path = default_registry_path(model_path)
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    registry_dir = os.path.dirname(registry_path) or "."
    data = _read_json(registry_path)

    if "engines" not in data:
        entry = _entry_from_manifest(data, registry_dir)
        return [entry] if entry else []

    raw_entries = data.get("engines")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Registry 'engines' must be a list: {registry_path}")

    entries = [
        entry
        for item in raw_entries
        if (entry := _entry_from_registry_item(item, registry_dir)) is not None
    ]
    return entries


def select_engine_for_video(
    entries: list[EngineRegistryEntry],
    video_info: VideoInfo,
    *,
    precision: str | None = None,
    io_precision: str | None = None,
) -> EngineRegistryEntry | None:
    """Select a static engine whose input shape matches the input video resolution."""
    matches: list[EngineRegistryEntry] = []
    for entry in entries:
        if precision is not None and entry.precision != precision:
            continue
        if io_precision is not None and entry.io_precision != io_precision:
            continue
        if entry.input_resolution != (video_info.width, video_info.height):
            continue
        if not os.path.exists(entry.engine_path):
            continue
        matches.append(entry)

    if not matches:
        return None

    # Prefer fp16 engines when the caller did not request a specific precision.
    return sorted(matches, key=lambda entry: entry.precision != "fp16")[0]


def format_registry_entries(entries: list[EngineRegistryEntry]) -> str:
    """Return a compact human-readable registry listing."""
    if not entries:
        return "  (no engines found)"

    lines: list[str] = []
    for entry in entries:
        input_res = entry.input_resolution
        output_res = entry.output_resolution
        input_text = f"{input_res[0]}x{input_res[1]}" if input_res else "dynamic/unknown"
        output_text = f"{output_res[0]}x{output_res[1]}" if output_res else "dynamic/unknown"
        precision = entry.precision or "unknown"
        io_precision = entry.io_precision or "unknown"
        exists = "ok" if os.path.exists(entry.engine_path) else "missing"
        lines.append(
            f"  {input_text} -> {output_text} | {precision} | io {io_precision} | "
            f"{exists} | {entry.engine_path}"
        )
    return "\n".join(lines)
