"""Full-decode and media-contract evidence for model compatibility reports."""

from __future__ import annotations

import json
import math
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from trtvideo.compatibility.evidence import file_identity
from trtvideo.models.manifest import ModelSpec
from trtvideo.video.color import ColorContractError, SdrColorContract
from trtvideo.video.metadata import VideoMetadata


class CompatibilityMediaError(RuntimeError):
    """Raised when FFmpeg cannot inspect compatibility media."""


@dataclass(frozen=True)
class MediaInspection:
    """Raw FFmpeg observations retained only during report generation."""

    identity: dict[str, Any]
    probe: dict[str, Any]
    packets: tuple[dict[str, Any], ...]
    decode_error: str | None


def _sanitize_tool_detail(detail: str, path: Path) -> str:
    sanitized = detail.replace(str(path), path.name)
    with suppress(OSError):
        sanitized = sanitized.replace(str(path.resolve()), path.name)
    sanitized = re.sub(r"/(?:home|Users)/[^/\s]+", "<user-home>", sanitized)
    return re.sub(r"/root(?:/|\b)", "<root-home>/", sanitized)


def _run_json(command: list[str], *, private_path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise CompatibilityMediaError(f"Cannot run {command[0]}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        detail = _sanitize_tool_detail(detail, private_path)
        raise CompatibilityMediaError(f"{command[0]} inspection failed: {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CompatibilityMediaError(f"{command[0]} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CompatibilityMediaError(f"{command[0]} returned a non-object JSON value")
    return value


def inspect_media(
    path: Path,
    *,
    collect_packets: bool,
    full_decode: bool,
) -> MediaInspection:
    """Inspect one media file without retaining its filesystem path."""
    identity = file_identity(path)
    probe = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        private_path=path,
    )

    packets: tuple[dict[str, Any], ...] = ()
    if collect_packets:
        packet_probe = _run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts,dts,pts_time,dts_time,duration_time,flags",
                "-of",
                "json",
                str(path),
            ],
            private_path=path,
        )
        raw_packets = packet_probe.get("packets", [])
        if not isinstance(raw_packets, list) or any(
            not isinstance(packet, dict) for packet in raw_packets
        ):
            raise CompatibilityMediaError("ffprobe packets field is invalid")
        packets = tuple(raw_packets)

    decode_error: str | None = None
    if full_decode:
        try:
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            decode_error = f"Cannot run ffmpeg: {type(exc).__name__}"
        else:
            if decoded.returncode != 0:
                detail = decoded.stderr.strip() or f"exit code {decoded.returncode}"
                decode_error = _sanitize_tool_detail(detail, path)

    return MediaInspection(
        identity=identity,
        probe=probe,
        packets=packets,
        decode_error=decode_error,
    )


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _streams(inspection: MediaInspection) -> list[dict[str, Any]]:
    streams = inspection.probe.get("streams", [])
    if not isinstance(streams, list):
        return []
    return [stream for stream in streams if isinstance(stream, dict)]


def _primary_video(inspection: MediaInspection, label: str) -> dict[str, Any]:
    videos = [stream for stream in _streams(inspection) if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise CompatibilityMediaError(f"{label} must contain exactly one video stream")
    return videos[0]


def _fps(stream: dict[str, Any]) -> Fraction | None:
    value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _frame_count(stream: dict[str, Any]) -> int | None:
    return _parse_int(stream.get("nb_read_frames") or stream.get("nb_frames"))


def _strictly_increasing(packets: tuple[dict[str, Any], ...], key: str) -> bool:
    previous: int | None = None
    for packet in packets:
        current = _parse_int(packet.get(key))
        if current is None or (previous is not None and current <= previous):
            return False
        previous = current
    return bool(packets)


def _nonvideo_inventory(inspection: MediaInspection) -> list[dict[str, Any]]:
    inventory = []
    for stream in _streams(inspection):
        stream_type = stream.get("codec_type")
        if stream_type == "video":
            continue
        raw_tags = stream.get("tags")
        tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else {}
        inventory.append(
            {
                "type": stream_type,
                "codec": stream.get("codec_name"),
                "codec_tag": stream.get("codec_tag_string"),
                "language": tags.get("language"),
            }
        )
    return sorted(inventory, key=lambda item: json.dumps(item, sort_keys=True))


def _format_bitrate(inspection: MediaInspection) -> int | None:
    payload = inspection.probe.get("format", {})
    if not isinstance(payload, dict):
        return None
    return _parse_int(payload.get("bit_rate"))


def _video_observation(
    stream: dict[str, Any],
    inspection: MediaInspection,
) -> dict[str, Any]:
    rate = _fps(stream)
    return {
        "codec": stream.get("codec_name"),
        "width": _parse_int(stream.get("width")),
        "height": _parse_int(stream.get("height")),
        "pixel_format": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "fps": str(rate) if rate is not None else None,
        "frame_count": _frame_count(stream),
        "duration_sec": _parse_float(stream.get("duration")),
        "video_bitrate_bps": _parse_int(stream.get("bit_rate")),
        "container_bitrate_bps": _format_bitrate(inspection),
        "has_b_frames": _parse_int(stream.get("has_b_frames")),
    }


def validate_media(
    input_inspection: MediaInspection,
    output_inspection: MediaInspection,
    *,
    model: ModelSpec,
    expected_frames: int,
) -> dict[str, Any]:
    """Validate smoke output against input media and the static engine contract."""
    if expected_frames <= 0:
        raise CompatibilityMediaError("expected_frames must be greater than zero")
    input_stream = _primary_video(input_inspection, "Input")
    output_stream = _primary_video(output_inspection, "Output")
    input_observed = _video_observation(input_stream, input_inspection)
    output_observed = _video_observation(output_stream, output_inspection)
    errors: list[str] = []

    input_shape = model.inputs[0].shape
    output_shape = model.outputs[0].shape
    expected_values = {
        "input width": (input_observed["width"], input_shape[3]),
        "input height": (input_observed["height"], input_shape[2]),
        "output width": (output_observed["width"], output_shape[3]),
        "output height": (output_observed["height"], output_shape[2]),
        "output frame count": (output_observed["frame_count"], expected_frames),
        "output packet count": (len(output_inspection.packets), expected_frames),
        "output B-frame depth": (output_observed["has_b_frames"], 0),
    }
    for label, (actual, expected) in expected_values.items():
        if actual != expected:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")

    input_frames = input_observed["frame_count"]
    if isinstance(input_frames, int) and input_frames < expected_frames:
        errors.append(f"input frame count: expected at least {expected_frames}, got {input_frames}")
    if input_observed["pixel_format"] not in {"nv12", "yuv420p"}:
        errors.append(f"input pixel format is unsupported: {input_observed['pixel_format']!r}")
    if output_observed["pixel_format"] != "yuv420p":
        errors.append(
            f"output pixel format: expected 'yuv420p', got {output_observed['pixel_format']!r}"
        )
    if output_observed["codec"] not in {"h264", "hevc"}:
        errors.append(f"output codec is unsupported: {output_observed['codec']!r}")
    output_bitrate = (
        output_observed["video_bitrate_bps"] or output_observed["container_bitrate_bps"]
    )
    if not isinstance(output_bitrate, int) or output_bitrate <= 0:
        errors.append("output bitrate is missing or non-positive")

    input_rate = _fps(input_stream)
    output_rate = _fps(output_stream)
    if input_rate is None or output_rate != input_rate:
        errors.append(f"output FPS: expected {input_rate!s}, got {output_rate!s}")
    if not _strictly_increasing(output_inspection.packets, "pts"):
        errors.append("output video PTS are not strictly increasing")
    if not _strictly_increasing(output_inspection.packets, "dts"):
        errors.append("output video DTS are not strictly increasing")
    if output_inspection.decode_error is not None:
        errors.append(f"full output decode failed: {output_inspection.decode_error}")

    if input_rate is not None:
        duration = output_observed["duration_sec"]
        if duration is None:
            format_payload = output_inspection.probe.get("format", {})
            if not isinstance(format_payload, dict):
                format_payload = {}
            duration = _parse_float(format_payload.get("duration"))
        expected_duration = expected_frames / float(input_rate)
        tolerance = 1.0 / float(input_rate) + 1e-6
        if duration is None or abs(duration - expected_duration) > tolerance:
            errors.append(
                "output duration: expected "
                f"{expected_duration:.6f}s +/- {tolerance:.6f}s, got {duration!r}"
            )

    metadata = VideoMetadata(
        width=int(input_observed["width"] or 0),
        height=int(input_observed["height"] or 0),
        fps=float(input_rate or 0),
        fps_str=str(input_rate or "0/1"),
        nb_frames=int(input_frames or 0),
        pix_fmt=input_observed["pixel_format"],
        color_range=input_observed["color_range"],
        color_space=input_observed["color_space"],
        color_transfer=input_observed["color_transfer"],
        color_primaries=input_observed["color_primaries"],
    )
    try:
        color = SdrColorContract.from_video_info(metadata)
    except ColorContractError as exc:
        errors.append(str(exc))
    else:
        expected_color = {
            "color_range": color.color_range,
            "color_space": color.color_space,
            "color_transfer": color.color_transfer,
            "color_primaries": color.color_primaries,
        }
        for field, expected_color_value in expected_color.items():
            if output_observed[field] != expected_color_value:
                errors.append(
                    "output "
                    f"{field}: expected {expected_color_value!r}, "
                    f"got {output_observed[field]!r}"
                )

    input_nonvideo = _nonvideo_inventory(input_inspection)
    output_nonvideo = _nonvideo_inventory(output_inspection)
    if output_nonvideo != input_nonvideo:
        errors.append("non-video stream inventory was not preserved")

    chapters = input_inspection.probe.get("chapters", [])
    output_chapters = output_inspection.probe.get("chapters", [])
    if (
        isinstance(input_frames, int)
        and expected_frames == input_frames
        and output_chapters != chapters
    ):
        errors.append("chapter inventory was not preserved for a full-length run")

    return {
        "valid": not errors,
        "expected_frames": expected_frames,
        "input": {
            "file": input_inspection.identity,
            "video": input_observed,
            "nonvideo_streams": input_nonvideo,
            "chapter_count": len(chapters) if isinstance(chapters, list) else None,
        },
        "output": {
            "file": output_inspection.identity,
            "video": output_observed,
            "nonvideo_streams": output_nonvideo,
            "chapter_count": (len(output_chapters) if isinstance(output_chapters, list) else None),
            "packet_count": len(output_inspection.packets),
            "pts_monotonic": _strictly_increasing(output_inspection.packets, "pts"),
            "dts_monotonic": _strictly_increasing(output_inspection.packets, "dts"),
            "full_decode": output_inspection.decode_error is None,
        },
        "errors": errors,
    }
