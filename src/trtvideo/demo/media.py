"""Live-action input preparation and output validation for the GPU demo."""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from trtvideo.demo import DemoError
from trtvideo.demo.config import (
    DEMO_COLOR_FRAME_INDEX,
    DEMO_FPS,
    DEMO_FRAMES,
    DEMO_MIN_AUDIO_MEAN_DBFS,
    DEMO_MIN_CHROMA_RETENTION_RATIO,
    VIDEO_AUTHOR,
    VIDEO_DURATION_SECONDS,
    VIDEO_LICENSE,
    VIDEO_LICENSE_URL,
    VIDEO_MODIFICATIONS,
    VIDEO_NAME,
    VIDEO_SOURCE_PAGE_URL,
    VIDEO_START_SECONDS,
    DemoPaths,
    DemoVideoContract,
)

_VIDEO_FILTER = (
    f"fps=fps={DEMO_FPS}:round=near,"
    "scale=1280:720:flags=lanczos,setsar=1,"
    "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
)
_X264_PARAMS = (
    "keyint=24:min-keyint=24:scenecut=0:bframes=0:"
    "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"
)
_BT709_OUTPUT_ARGS = (
    "-color_range",
    "tv",
    "-colorspace",
    "bt709",
    "-color_trc",
    "bt709",
    "-color_primaries",
    "bt709",
)
_VOLUME_PATTERN = re.compile(r"(mean|max)_volume:\s+(-?(?:inf|\d+(?:\.\d+)?)) dB")


def build_demo_input_command(paths: DemoPaths) -> list[str]:
    """Prepare the pinned excerpt for the static demo engine."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    command += ["-ss", str(VIDEO_START_SECONDS), "-i", str(paths.source_video)]
    command += ["-map", "0:v:0", "-map", "0:a:0"]
    command += [
        "-vf",
        _VIDEO_FILTER,
        "-frames:v",
        str(DEMO_FRAMES),
        "-t",
        str(VIDEO_DURATION_SECONDS),
    ]
    command += ["-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p"]
    command += ["-x264-params", _X264_PARAMS, *_BT709_OUTPUT_ARGS]
    command += ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"]
    command += [
        "-metadata",
        f"title={VIDEO_NAME}",
        "-metadata",
        f"artist={VIDEO_AUTHOR}",
        "-metadata",
        f"comment={VIDEO_MODIFICATIONS}",
        "-metadata",
        f"copyright={VIDEO_LICENSE} ({VIDEO_LICENSE_URL}); source: {VIDEO_SOURCE_PAGE_URL}",
    ]
    command.append(str(paths.input_video))
    return command


def _run_json(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise DemoError(f"cannot inspect demo output: {shlex.join(command)}") from exc
    if not isinstance(value, dict):
        raise DemoError("ffprobe returned a non-object JSON value")
    return value


def _probe_video(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probe = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    packet_probe = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,flags",
            "-of",
            "json",
            str(path),
        ]
    )
    packets = packet_probe.get("packets", [])
    if not isinstance(packets, list):
        raise DemoError("ffprobe packets field is not a list")
    return probe, packets


def _probe_chroma(path: Path) -> dict[str, float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        (f"select=eq(n\\,{DEMO_COLOR_FRAME_INDEX}),signalstats,metadata=print:file=-"),
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoError(f"cannot inspect demo chroma: {shlex.join(command)}") from exc

    wanted = {"ULOW", "UHIGH", "VLOW", "VHIGH"}
    observed: dict[str, float] = {}
    prefix = "lavfi.signalstats."
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        if name not in wanted:
            continue
        try:
            observed[name] = float(value)
        except ValueError as exc:
            raise DemoError(f"invalid FFmpeg signalstats value: {line}") from exc
    if observed.keys() != wanted:
        missing = ", ".join(sorted(wanted - observed.keys()))
        raise DemoError(f"FFmpeg signalstats omitted demo chroma fields: {missing}")
    return observed


def _probe_audio_levels(path: Path) -> dict[str, float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoError(f"cannot inspect demo audio: {shlex.join(command)}") from exc

    levels = {name: float(value) for name, value in _VOLUME_PATTERN.findall(result.stderr)}
    if levels.keys() != {"mean", "max"}:
        raise DemoError("FFmpeg volumedetect omitted demo audio levels")
    if levels["mean"] < DEMO_MIN_AUDIO_MEAN_DBFS:
        raise DemoError(
            "demo audio is too quiet: expected mean >= "
            f"{DEMO_MIN_AUDIO_MEAN_DBFS:.1f} dBFS, got {levels['mean']:.1f} dBFS"
        )
    return {
        "mean_dbfs": levels["mean"],
        "peak_dbfs": levels["max"],
    }


def summarize_demo_chroma(stats: dict[str, float]) -> dict[str, float | int]:
    """Summarize chroma percentiles in the pinned real-video frame."""
    u_span = stats["UHIGH"] - stats["ULOW"]
    v_span = stats["VHIGH"] - stats["VLOW"]
    if not all(math.isfinite(value) and value > 0 for value in (u_span, v_span)):
        raise DemoError(f"demo chroma spans must be positive, got U={u_span:.1f}, V={v_span:.1f}")
    return {
        "frame_index": DEMO_COLOR_FRAME_INDEX,
        "u_percentile_span": u_span,
        "v_percentile_span": v_span,
    }


def validate_demo_color_preservation(
    input_chroma: dict[str, float | int],
    output_chroma: dict[str, float | int],
) -> dict[str, float]:
    """Reject severe chroma collapse relative to the pinned live-action input."""
    ratios = {}
    for plane in ("u", "v"):
        key = f"{plane}_percentile_span"
        input_span = float(input_chroma[key])
        output_span = float(output_chroma[key])
        if input_span <= 0:
            raise DemoError(f"demo input {plane.upper()} chroma span must be positive")
        ratios[f"{plane}_retention_ratio"] = output_span / input_span
    if min(ratios.values()) < DEMO_MIN_CHROMA_RETENTION_RATIO:
        raise DemoError(
            "demo chroma preservation failed: expected U/V retention >= "
            f"{DEMO_MIN_CHROMA_RETENTION_RATIO:.2f}, got "
            f"U={ratios['u_retention_ratio']:.3f}, V={ratios['v_retention_ratio']:.3f}"
        )
    return ratios


def _strictly_increasing(packets: list[dict[str, Any]], key: str) -> bool:
    previous: int | None = None
    for packet in packets:
        try:
            current = int(packet[key])
        except (KeyError, TypeError, ValueError):
            return False
        if previous is not None and current <= previous:
            return False
        previous = current
    return True


def _stream_inventory(
    streams: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, list[dict[str, Any]]]:
    by_type = {
        stream_type: [stream for stream in streams if stream.get("codec_type") == stream_type]
        for stream_type in ("video", "audio")
    }
    expected_counts = {"video": 1, "audio": 1}
    for stream_type, expected_count in expected_counts.items():
        if len(by_type[stream_type]) != expected_count:
            errors.append(
                f"{stream_type} stream count: expected {expected_count}, "
                f"got {len(by_type[stream_type])}"
            )
    return by_type


def _validate_video_stream(
    video: dict[str, Any],
    packets: list[dict[str, Any]],
    contract: DemoVideoContract,
    errors: list[str],
) -> None:
    expected_video: dict[str, object] = {
        "codec_name": "h264",
        "width": contract.width,
        "height": contract.height,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "has_b_frames": 0,
    }
    for key, expected_value in expected_video.items():
        if video.get(key) != expected_value:
            errors.append(f"{key}: expected {expected_value!r}, got {video.get(key)!r}")

    try:
        actual_fps = Fraction(str(video.get("avg_frame_rate") or video.get("r_frame_rate")))
    except (TypeError, ValueError, ZeroDivisionError):
        actual_fps = Fraction(0, 1)
    if actual_fps != Fraction(contract.fps):
        errors.append(f"fps: expected {contract.fps}, got {actual_fps}")

    frame_count = video.get("nb_read_frames") or video.get("nb_frames")
    if str(frame_count) != str(contract.frames):
        errors.append(f"frame count: expected {contract.frames}, got {frame_count}")
    if len(packets) != contract.frames:
        errors.append(f"packet count: expected {contract.frames}, got {len(packets)}")
    if not _strictly_increasing(packets, "pts"):
        errors.append("video PTS are not strictly increasing")
    if not _strictly_increasing(packets, "dts"):
        errors.append("video DTS are not strictly increasing")
    if not packets or "K" not in str(packets[0].get("flags", "")):
        errors.append("first video packet is not a keyframe")


def validate_demo_probe(
    probe: dict[str, Any],
    packets: list[dict[str, Any]],
    contract: DemoVideoContract,
) -> dict[str, Any]:
    """Validate the demo video and source-audio contract from FFprobe data."""
    errors: list[str] = []
    streams = probe.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    by_type = _stream_inventory(streams, errors)
    video = by_type["video"][0] if len(by_type["video"]) == 1 else {}
    _validate_video_stream(video, packets, contract, errors)

    if errors:
        raise DemoError("demo media validation failed:\n- " + "\n- ".join(errors))
    return {
        "width": contract.width,
        "height": contract.height,
        "fps": contract.fps,
        "frames": contract.frames,
        "stream_counts": {
            "video": 1,
            "audio": 1,
        },
        "timestamps": "strictly_monotonic",
    }


def validate_demo_video(path: Path, contract: DemoVideoContract) -> dict[str, Any]:
    """Fully decode and validate one demo input/output."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v",
                "-map",
                "0:a?",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoError(f"full decode failed for {path}") from exc
    probe, packets = _probe_video(path)
    observed = validate_demo_probe(probe, packets, contract)
    observed["audio"] = _probe_audio_levels(path)
    observed["chroma"] = summarize_demo_chroma(_probe_chroma(path))
    return observed
