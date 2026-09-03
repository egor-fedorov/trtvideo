"""Pinned and user-supplied input preparation for compatibility checks."""

from __future__ import annotations

import json
import math
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trtvideo.compatibility.evidence import (
    CompatibilityEvidenceError,
    file_identity,
    load_json_object,
    public_value,
    sha256_file,
)
from trtvideo.demo.config import (
    VIDEO_AUTHOR,
    VIDEO_LICENSE,
    VIDEO_LICENSE_URL,
    VIDEO_NAME,
    VIDEO_SHA256,
    VIDEO_SIZE_BYTES,
    VIDEO_SOURCE_PAGE_URL,
    VIDEO_URL,
)
from trtvideo.demo.media import build_live_action_input_command

INPUT_SCHEMA_VERSION = 1
FIXTURE_CONTRACT_VERSION = "jacqueville-live-action-sdr-v1"
DEFAULT_INPUT_WIDTH = 1280
DEFAULT_INPUT_HEIGHT = 720
DEFAULT_INPUT_FRAMES = 120
DEFAULT_INPUT_FPS = "24/1"
_UNKNOWN_COLOR_VALUES = {None, "", "unknown", "reserved"}


class CompatibilityInputError(RuntimeError):
    """Raised when a compatibility input cannot be prepared or validated."""


@dataclass(frozen=True)
class InputPreparation:
    """Complete request for one deterministic compatibility input."""

    output: Path
    manifest: Path
    width: int
    height: int
    frames: int = DEFAULT_INPUT_FRAMES
    source: Path | None = None
    source_cache: Path | None = None

    @property
    def uses_pinned_source(self) -> bool:
        return self.source is None


def _run_json(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise CompatibilityInputError(f"Media inspection failed: {command[0]}") from exc
    if not isinstance(payload, dict):
        raise CompatibilityInputError(f"{command[0]} returned a non-object JSON value")
    return payload


def probe_video_size(path: Path) -> tuple[int, int]:
    """Return the video size after checking the supported custom-input color contract."""
    if not path.is_file():
        raise CompatibilityInputError(f"Input video does not exist: {path}")
    payload = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,color_space,color_transfer,color_primaries",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise CompatibilityInputError("Input must contain exactly one primary video stream")
    try:
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompatibilityInputError("Input video has no valid dimensions") from exc
    if width <= 0 or height <= 0:
        raise CompatibilityInputError("Input video dimensions must be positive")
    stream = streams[0]
    transfer = stream.get("color_transfer")
    if transfer in {"smpte2084", "arib-std-b67"}:
        raise CompatibilityInputError(
            "Custom input uses an HDR transfer function; convert/tonemap it to SDR BT.709 first"
        )
    default_space = "bt709" if width >= 1280 or height >= 720 else "smpte170m"
    unsupported = {
        field: value
        for field in ("color_space", "color_transfer", "color_primaries")
        if (value := stream.get(field)) not in _UNKNOWN_COLOR_VALUES and value != "bt709"
    }
    if not unsupported and default_space != "bt709":
        unknown = [
            field
            for field in ("color_space", "color_transfer", "color_primaries")
            if stream.get(field) in _UNKNOWN_COLOR_VALUES
        ]
        if unknown:
            unsupported["missing"] = ",".join(unknown)
    if unsupported:
        details = ", ".join(f"{field}={value}" for field, value in unsupported.items())
        raise CompatibilityInputError(
            f"Custom input must declare SDR BT.709 color metadata ({details}); "
            "convert it to BT.709 first"
        )
    return width, height


def _verify_pinned_source(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == VIDEO_SIZE_BYTES
        and sha256_file(path) == VIDEO_SHA256
    )


def download_pinned_source(path: Path) -> None:
    """Download and hash-check the immutable CC BY-SA live-action source."""
    if _verify_pinned_source(path):
        print(f"Using verified live-action source: {path}", flush=True)
        return
    if path.exists():
        raise CompatibilityInputError(
            f"Pinned source cache is invalid: {path}; remove it before retrying"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "trtvideo/compatibility"})
    print(f"Downloading pinned live-action source ({VIDEO_SIZE_BYTES / 1e6:.1f} MB)...", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise CompatibilityInputError(f"Pinned source download failed: {exc}") from exc
    if not _verify_pinned_source(partial):
        partial.unlink(missing_ok=True)
        raise CompatibilityInputError("Downloaded source failed size/SHA256 verification")
    os.replace(partial, path)


def input_command(request: InputPreparation) -> list[str]:
    """Build the exact FFmpeg command represented by an input manifest."""
    source = request.source or request.source_cache
    if source is None:
        raise CompatibilityInputError("A source cache path is required for the pinned fixture")
    return build_live_action_input_command(
        source=source,
        output=request.output,
        width=request.width,
        height=request.height,
        frames=request.frames,
        start_seconds=14 if request.uses_pinned_source else 0,
        include_attribution=request.uses_pinned_source,
        require_audio=request.uses_pinned_source,
    )


def _validate_prepared_video(path: Path, request: InputPreparation) -> dict[str, Any]:
    probe = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise CompatibilityInputError("Prepared input has no stream inventory")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise CompatibilityInputError("Prepared input must contain exactly one video stream")
    video = videos[0]
    expected = {
        "codec_name": "h264",
        "width": request.width,
        "height": request.height,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "has_b_frames": 0,
        "avg_frame_rate": DEFAULT_INPUT_FPS,
        "nb_read_frames": str(request.frames),
    }
    errors = [
        f"{field}: expected {wanted!r}, got {video.get(field)!r}"
        for field, wanted in expected.items()
        if video.get(field) != wanted
    ]
    if request.uses_pinned_source and len(audios) != 1:
        errors.append(f"audio stream count: expected 1, got {len(audios)}")
    packet_payload = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts",
            "-of",
            "json",
            str(path),
        ]
    )
    packets = packet_payload.get("packets")
    if not isinstance(packets, list) or len(packets) != request.frames:
        errors.append(
            f"video packet count: expected {request.frames}, "
            f"got {len(packets) if isinstance(packets, list) else 'invalid'}"
        )
        packets = []
    for field in ("pts", "dts"):
        try:
            values = [int(packet[field]) for packet in packets]
        except (KeyError, TypeError, ValueError):
            values = []
        if len(values) != request.frames or any(
            current <= previous for previous, current in zip(values, values[1:], strict=False)
        ):
            errors.append(f"video {field.upper()} are not strictly increasing")
    expected_duration = request.frames / 24
    try:
        observed_duration = float(probe.get("format", {}).get("duration"))
    except (AttributeError, TypeError, ValueError):
        observed_duration = math.nan
    if not math.isfinite(observed_duration) or abs(observed_duration - expected_duration) > 1 / 24:
        errors.append(
            f"duration: expected {expected_duration:.6f}s +/- {1 / 24:.6f}s, "
            f"got {observed_duration!r}"
        )
    if request.uses_pinned_source:
        format_payload = probe.get("format", {})
        tags = format_payload.get("tags", {}) if isinstance(format_payload, dict) else {}
        normalized_tags = (
            {str(key).lower(): str(value) for key, value in tags.items()}
            if isinstance(tags, dict)
            else {}
        )
        if normalized_tags.get("title") != VIDEO_NAME:
            errors.append("pinned input title attribution is missing")
        if normalized_tags.get("artist") != VIDEO_AUTHOR:
            errors.append("pinned input author attribution is missing")
        copyright_value = normalized_tags.get("copyright", "")
        if VIDEO_LICENSE not in copyright_value or VIDEO_SOURCE_PAGE_URL not in copyright_value:
            errors.append("pinned input license/source attribution is missing")
    try:
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise CompatibilityInputError("Cannot run ffmpeg for full input decode") from exc
    if decoded.returncode != 0:
        errors.append("full input decode failed")
    if errors:
        raise CompatibilityInputError("Prepared input validation failed:\n- " + "\n- ".join(errors))
    return {
        "width": request.width,
        "height": request.height,
        "frames": request.frames,
        "fps": DEFAULT_INPUT_FPS,
        "audio_streams": len(audios),
        "duration_sec": observed_duration,
        "timestamps": "strictly_monotonic",
        "full_decode": True,
    }


def prepare_input(request: InputPreparation) -> dict[str, Any]:
    """Create and validate one input, then atomically record its contract."""
    if request.width <= 0 or request.height <= 0 or request.frames <= 0:
        raise CompatibilityInputError("Width, height, and frame count must be positive")
    source = request.source
    if request.uses_pinned_source:
        if request.source_cache is None:
            raise CompatibilityInputError("Pinned input preparation requires --source-cache")
        download_pinned_source(request.source_cache)
        source = request.source_cache
    elif source is None or not source.is_file():
        raise CompatibilityInputError(f"Input video does not exist: {source}")
    else:
        probe_video_size(source)

    assert source is not None
    try:
        same_file = source.resolve() == request.output.resolve()
    except OSError:
        same_file = source == request.output
    if same_file:
        raise CompatibilityInputError("Source and prepared output must be different files")

    request.output.parent.mkdir(parents=True, exist_ok=True)
    request.manifest.parent.mkdir(parents=True, exist_ok=True)
    request.output.unlink(missing_ok=True)
    request.manifest.unlink(missing_ok=True)
    command = input_command(request)
    print("Preparing compatibility input with FFmpeg...", flush=True)
    try:
        subprocess.run(command, check=True)
        observed = _validate_prepared_video(request.output, request)
    except CompatibilityInputError:
        request.output.unlink(missing_ok=True)
        raise
    except (OSError, subprocess.CalledProcessError):
        request.output.unlink(missing_ok=True)
        raise CompatibilityInputError("FFmpeg input preparation failed") from None

    manifest = {
        "document_type": "trtvideo-compatibility-input",
        "schema_version": INPUT_SCHEMA_VERSION,
        "fixture_contract": FIXTURE_CONTRACT_VERSION,
        "source_kind": "pinned-live-action" if request.uses_pinned_source else "user-supplied",
        "source": file_identity(source),
        "output": file_identity(request.output),
        "command": command,
        "observed": observed,
    }
    if request.uses_pinned_source:
        manifest["attribution"] = {
            "name": VIDEO_NAME,
            "author": VIDEO_AUTHOR,
            "source": VIDEO_SOURCE_PAGE_URL,
            "license": VIDEO_LICENSE,
            "license_url": VIDEO_LICENSE_URL,
        }
    temporary = request.manifest.with_name(f".{request.manifest.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, request.manifest)
    return manifest


def input_manifest_evidence(path: Path, prepared_input: Path) -> dict[str, Any]:
    """Validate and sanitize input-preparation evidence for a public report."""
    manifest = load_json_object(path, "Input-preparation manifest")
    if manifest.get("document_type") != "trtvideo-compatibility-input":
        raise CompatibilityEvidenceError("Input-preparation document_type is invalid")
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise CompatibilityEvidenceError("Input-preparation schema_version is unsupported")

    prepared_identity = file_identity(prepared_input)
    if manifest.get("output") != prepared_identity:
        raise CompatibilityEvidenceError(
            "Prepared input identity does not match its input-preparation manifest"
        )

    source_kind = manifest.get("source_kind")
    if source_kind not in {"pinned-live-action", "user-supplied"}:
        raise CompatibilityEvidenceError("Input-preparation source_kind is invalid")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise CompatibilityEvidenceError("Input-preparation source identity is missing")
    try:
        raw_source_name = source["name"]
        raw_source_sha256 = source["sha256"]
        raw_source_size = source["size_bytes"]
    except KeyError as exc:
        raise CompatibilityEvidenceError("Input-preparation source identity is invalid") from exc
    if not isinstance(raw_source_name, str) or not isinstance(raw_source_sha256, str):
        raise CompatibilityEvidenceError("Input-preparation source identity is invalid")
    if not isinstance(raw_source_size, int) or isinstance(raw_source_size, bool):
        raise CompatibilityEvidenceError("Input-preparation source identity is invalid")
    source_name = public_value(raw_source_name, "Input source name")
    source_sha256 = raw_source_sha256
    source_size = raw_source_size
    if Path(source_name).name != source_name or source_size <= 0:
        raise CompatibilityEvidenceError("Input-preparation source identity is invalid")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise CompatibilityEvidenceError("Input-preparation source SHA256 is invalid")

    observed = manifest.get("observed")
    if not isinstance(observed, dict):
        raise CompatibilityEvidenceError("Input-preparation observations are missing")
    integer_fields = ("width", "height", "frames", "audio_streams")
    if any(
        not isinstance(observed.get(field), int)
        or isinstance(observed.get(field), bool)
        or int(observed[field]) < (0 if field == "audio_streams" else 1)
        for field in integer_fields
    ):
        raise CompatibilityEvidenceError("Input-preparation observations are invalid")
    duration = observed.get("duration_sec")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) <= 0
        or observed.get("fps") != DEFAULT_INPUT_FPS
        or observed.get("timestamps") != "strictly_monotonic"
        or observed.get("full_decode") is not True
    ):
        raise CompatibilityEvidenceError("Input-preparation observations are invalid")
    safe_observed = {
        "width": observed["width"],
        "height": observed["height"],
        "frames": observed["frames"],
        "fps": DEFAULT_INPUT_FPS,
        "audio_streams": observed["audio_streams"],
        "duration_sec": float(duration),
        "timestamps": "strictly_monotonic",
        "full_decode": True,
    }

    fixture_contract = manifest.get("fixture_contract")
    if fixture_contract != FIXTURE_CONTRACT_VERSION:
        raise CompatibilityEvidenceError("Input fixture contract is unsupported")
    attribution: dict[str, str] | None = None
    if source_kind == "pinned-live-action":
        if source_sha256 != VIDEO_SHA256 or source_size != VIDEO_SIZE_BYTES:
            raise CompatibilityEvidenceError("Pinned input source identity is invalid")
        expected_attribution = {
            "name": VIDEO_NAME,
            "author": VIDEO_AUTHOR,
            "source": VIDEO_SOURCE_PAGE_URL,
            "license": VIDEO_LICENSE,
            "license_url": VIDEO_LICENSE_URL,
        }
        if manifest.get("attribution") != expected_attribution:
            raise CompatibilityEvidenceError("Pinned input attribution is invalid")
        attribution = expected_attribution

    return {
        "manifest": file_identity(path),
        "fixture_contract": fixture_contract,
        "source_kind": source_kind,
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "size_bytes": source_size,
        },
        "prepared_input": prepared_identity,
        "observed": safe_observed,
        "attribution": attribution,
    }
