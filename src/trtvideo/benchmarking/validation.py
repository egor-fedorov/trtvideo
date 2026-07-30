"""FFmpeg-based validation for benchmark video outputs."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


class OutputValidationError(RuntimeError):
    """Raised when FFmpeg cannot inspect a benchmark output."""


@dataclass(frozen=True)
class OutputContract:
    """Expected media properties for one benchmark output."""

    width: int
    height: int
    fps: str
    frames: int
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    color_range: str = "tv"
    color_space: str = "bt709"
    color_transfer: str = "bt709"
    color_primaries: str = "bt709"
    has_b_frames: int | None = 0
    gop_frames: int | None = None
    target_bitrate_mbps: float | None = None
    bitrate_tolerance: float = 0.10
    require_monotonic_pts: bool = True
    require_monotonic_dts: bool = True


def _run_json_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise OutputValidationError(f"Command failed: {' '.join(command)}: {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"Invalid JSON from {' '.join(command)}: {exc}") from exc
    if not isinstance(value, dict):
        raise OutputValidationError(f"Expected JSON object from {' '.join(command)}")
    return value


def probe_output(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Decode and inspect an output without including validation in timed work."""
    decode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    decode = subprocess.run(decode_command, capture_output=True, text=True, check=False)
    decode_error = decode.stderr.strip() or None
    if decode.returncode != 0 and decode_error is None:
        decode_error = f"ffmpeg decode exited with code {decode.returncode}"

    probe = _run_json_command(
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
    packet_probe = _run_json_command(
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
        ]
    )
    packets = packet_probe.get("packets", [])
    if not isinstance(packets, list):
        raise OutputValidationError("ffprobe packets field is not a list")
    return probe, packets, decode_error


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


def _check_equal(
    checks: dict[str, bool],
    errors: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    passed = actual == expected
    checks[name] = passed
    if not passed:
        errors.append(f"{name}: expected {expected!r}, got {actual!r}")


def _timestamps_are_monotonic(
    packets: list[dict[str, Any]],
    key: str,
) -> tuple[bool, int | None]:
    previous: int | None = None
    for index, packet in enumerate(packets):
        value = _parse_int(packet.get(key))
        if value is None:
            return False, index
        if previous is not None and value <= previous:
            return False, index
        previous = value
    return True, None


def validate_output_probe(
    probe: dict[str, Any],
    packets: list[dict[str, Any]],
    *,
    contract: OutputContract,
    decode_error: str | None = None,
) -> dict[str, Any]:
    """Validate parsed FFmpeg metadata against a benchmark output contract."""
    errors: list[str] = []
    checks: dict[str, bool] = {}
    streams = probe.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    other_streams = [stream for stream in streams if stream.get("codec_type") != "video"]

    checks["full_decode"] = decode_error is None
    if decode_error is not None:
        errors.append(f"full_decode: {decode_error}")
    _check_equal(checks, errors, "video_stream_count", len(video_streams), 1)
    _check_equal(checks, errors, "other_stream_count", len(other_streams), 0)

    stream = video_streams[0] if len(video_streams) == 1 else {}
    _check_equal(checks, errors, "codec", stream.get("codec_name"), contract.codec)
    _check_equal(checks, errors, "width", _parse_int(stream.get("width")), contract.width)
    _check_equal(checks, errors, "height", _parse_int(stream.get("height")), contract.height)
    _check_equal(checks, errors, "pixel_format", stream.get("pix_fmt"), contract.pixel_format)
    _check_equal(checks, errors, "color_range", stream.get("color_range"), contract.color_range)
    _check_equal(checks, errors, "color_space", stream.get("color_space"), contract.color_space)
    _check_equal(
        checks,
        errors,
        "color_transfer",
        stream.get("color_transfer"),
        contract.color_transfer,
    )
    _check_equal(
        checks,
        errors,
        "color_primaries",
        stream.get("color_primaries"),
        contract.color_primaries,
    )
    if contract.has_b_frames is not None:
        _check_equal(
            checks,
            errors,
            "has_b_frames",
            _parse_int(stream.get("has_b_frames")),
            contract.has_b_frames,
        )

    actual_fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    try:
        fps_matches = Fraction(str(actual_fps)) == Fraction(contract.fps)
    except (ValueError, ZeroDivisionError):
        fps_matches = False
    checks["fps"] = fps_matches
    if not fps_matches:
        errors.append(f"fps: expected {contract.fps!r}, got {actual_fps!r}")

    frame_count = _parse_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _parse_int(stream.get("nb_frames"))
    _check_equal(checks, errors, "frame_count", frame_count, contract.frames)
    _check_equal(checks, errors, "packet_count", len(packets), contract.frames)

    duration = _parse_float(stream.get("duration"))
    if duration is None:
        duration = _parse_float(probe.get("format", {}).get("duration"))
    expected_duration = contract.frames / float(Fraction(contract.fps))
    duration_tolerance = 1.0 / float(Fraction(contract.fps)) + 1e-6
    duration_matches = (
        duration is not None and abs(duration - expected_duration) <= duration_tolerance
    )
    checks["duration"] = duration_matches
    if not duration_matches:
        errors.append(
            f"duration: expected {expected_duration:.6f}s +/- {duration_tolerance:.6f}s, "
            f"got {duration!r}"
        )

    pts_monotonic, bad_pts_index = _timestamps_are_monotonic(packets, "pts")
    dts_monotonic, bad_dts_index = _timestamps_are_monotonic(packets, "dts")
    if contract.require_monotonic_pts:
        checks["pts_monotonic"] = pts_monotonic
        if not pts_monotonic:
            errors.append(f"pts_monotonic: invalid packet at index {bad_pts_index}")
    if contract.require_monotonic_dts:
        checks["dts_monotonic"] = dts_monotonic
        if not dts_monotonic:
            errors.append(f"dts_monotonic: invalid packet at index {bad_dts_index}")

    keyframe_indices = [
        index for index, packet in enumerate(packets) if "K" in str(packet.get("flags", ""))
    ]
    first_keyframe = bool(keyframe_indices) and keyframe_indices[0] == 0
    checks["first_packet_keyframe"] = first_keyframe
    if not first_keyframe:
        errors.append("first_packet_keyframe: packet 0 is not a keyframe")

    keyframe_gaps = [
        current - previous
        for previous, current in zip(keyframe_indices, keyframe_indices[1:], strict=False)
    ]
    if contract.gop_frames is not None:
        expected_keyframes = max(1, math.ceil(contract.frames / contract.gop_frames))
        gop_matches = len(keyframe_indices) >= expected_keyframes and (
            expected_keyframes == 1
            or (
                bool(keyframe_gaps)
                and all(abs(gap - contract.gop_frames) <= 1 for gap in keyframe_gaps)
            )
        )
        checks["keyframe_interval"] = gop_matches
        if not gop_matches:
            errors.append(
                f"keyframe_interval: expected {contract.gop_frames} +/- 1 frames, "
                f"got {keyframe_gaps!r}"
            )

    stream_bitrate = _parse_int(stream.get("bit_rate"))
    container_bitrate = _parse_int(probe.get("format", {}).get("bit_rate"))
    actual_bitrate = stream_bitrate or container_bitrate
    bitrate_delta: float | None = None
    if contract.target_bitrate_mbps is not None:
        target_bitrate = contract.target_bitrate_mbps * 1_000_000
        if actual_bitrate is not None:
            bitrate_delta = abs(actual_bitrate - target_bitrate) / target_bitrate
        bitrate_matches = bitrate_delta is not None and bitrate_delta <= contract.bitrate_tolerance
        checks["bitrate"] = bitrate_matches
        if not bitrate_matches:
            errors.append(
                f"bitrate: expected {contract.target_bitrate_mbps:.3f} Mbps +/- "
                f"{contract.bitrate_tolerance * 100:.1f}%, got "
                f"{actual_bitrate / 1_000_000 if actual_bitrate else None!r} Mbps"
            )

    observed = {
        "codec": stream.get("codec_name"),
        "width": _parse_int(stream.get("width")),
        "height": _parse_int(stream.get("height")),
        "pixel_format": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "fps": actual_fps,
        "duration_sec": duration,
        "frame_count": frame_count,
        "packet_count": len(packets),
        "has_b_frames": _parse_int(stream.get("has_b_frames")),
        "keyframe_count": len(keyframe_indices),
        "keyframe_gaps_frames": keyframe_gaps,
        "video_bitrate_bps": actual_bitrate,
        "bitrate_relative_delta": bitrate_delta,
    }
    return {
        "valid": not errors,
        "contract": asdict(contract),
        "checks": checks,
        "observed": observed,
        "errors": errors,
    }


def validate_output(path: Path, contract: OutputContract) -> dict[str, Any]:
    """Run full decode/probe validation for a benchmark output."""
    try:
        probe, packets, decode_error = probe_output(path)
    except OutputValidationError as exc:
        return {
            "valid": False,
            "contract": asdict(contract),
            "checks": {},
            "observed": {},
            "errors": [str(exc)],
        }
    return validate_output_probe(
        probe,
        packets,
        contract=contract,
        decode_error=decode_error,
    )
