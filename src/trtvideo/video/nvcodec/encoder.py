"""NVENC settings shared by production and comparative benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from trtvideo.video.fps import parse_fps_fraction


def format_nvenc_fps(value: str) -> str:
    """Return a PyNvVideoCodec ``fps`` parameter without integer rounding."""
    fps = parse_fps_fraction(value)
    if fps <= 0:
        raise ValueError(f"invalid frame rate: {value}")
    if fps.denominator == 1:
        return str(fps.numerator)
    return str(float(fps))


def gop_size_for_one_second(value: str) -> int:
    """Return an NVENC GOP/IDR interval of roughly one second."""
    fps = parse_fps_fraction(value)
    if fps <= 0:
        raise ValueError(f"invalid frame rate: {value}")
    return max(1, int(fps + Fraction(1, 2)))


@dataclass(frozen=True)
class NvencCbrContract:
    """Exact single-pass NVENC CBR settings used by comparative benchmarks."""

    bitrate_bps: int
    gop_frames: int
    codec: str = "h264"

    def __post_init__(self) -> None:
        if self.bitrate_bps <= 0:
            raise ValueError("NVENC bitrate must be greater than zero")
        if self.gop_frames <= 0:
            raise ValueError("NVENC GOP must be greater than zero")
        if self.codec not in {"h264", "hevc"}:
            raise ValueError(f"Unsupported NVENC codec: {self.codec}")

    @property
    def vbv_buffer_bits(self) -> int:
        """Use a two-second VBV buffer at the target bitrate."""
        return self.bitrate_bps * 2

    @property
    def vbv_initial_delay_bits(self) -> int:
        """Start the two-second VBV buffer at 50% occupancy."""
        return self.bitrate_bps

    def pynvcodec_options(self) -> dict[str, int | str]:
        """Return PyNvVideoCodec kwargs for the common encoder contract."""
        return {
            "codec": self.codec,
            "bitrate": self.bitrate_bps,
            "maxbitrate": self.bitrate_bps,
            "vbvbufsize": self.vbv_buffer_bits,
            "vbvinit": self.vbv_initial_delay_bits,
            "preset": "P4",
            "tuning_info": "high_quality",
            "rc": "cbr",
            "gop": self.gop_frames,
            "idrperiod": self.gop_frames,
            "bf": 0,
            "repeatspspps": 1,
        }

    def ffmpeg_options(self) -> list[str]:
        """Return equivalent FFmpeg h264/hevc_nvenc arguments."""
        encoder = "h264_nvenc" if self.codec == "h264" else "hevc_nvenc"
        bitrate = str(self.bitrate_bps)
        return [
            "-c:v",
            encoder,
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "cbr",
            "-b:v",
            bitrate,
            "-minrate",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            str(self.vbv_buffer_bits),
            "-rc_init_occupancy",
            str(self.vbv_initial_delay_bits),
            "-multipass",
            "disabled",
            "-rc-lookahead",
            "0",
            "-spatial-aq",
            "0",
            "-temporal-aq",
            "0",
            "-bf",
            "0",
            "-g",
            str(self.gop_frames),
            "-forced-idr",
            "1",
        ]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the contract into benchmark manifests."""
        return {
            "codec": self.codec,
            "preset": "p4",
            "tuning": "high_quality",
            "rate_control": "cbr",
            "target_bitrate_bps": self.bitrate_bps,
            "min_bitrate_bps": self.bitrate_bps,
            "max_bitrate_bps": self.bitrate_bps,
            "vbv_buffer_bits": self.vbv_buffer_bits,
            "vbv_initial_delay_bits": self.vbv_initial_delay_bits,
            "multipass": "disabled",
            "lookahead_frames": 0,
            "spatial_aq": False,
            "temporal_aq": False,
            "gop_frames": self.gop_frames,
            "b_frames": 0,
        }
