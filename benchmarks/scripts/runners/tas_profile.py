"""I/O execution profiles for TheAnimeScripter video benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

from benchmarks.scripts.contracts.benchmark import CompetitorError

ExecutionProfileName = Literal["upstream-default", "tuned"]
DecodeMethod = Literal["cpu", "nvdec"]
Writer = Literal["ffmpeg", "nelux"]


@dataclass(frozen=True)
class TasExecutionProfile:
    """Effective upstream TAS I/O path; its TensorRT CUDA Graph stays enabled."""

    name: ExecutionProfileName
    decode_method: DecodeMethod
    writer: Writer

    def __post_init__(self) -> None:
        if self.name not in {"upstream-default", "tuned"}:
            raise CompetitorError(f"Unsupported TAS execution profile: {self.name}")
        if self.decode_method not in {"cpu", "nvdec"} or self.writer not in {"ffmpeg", "nelux"}:
            raise CompetitorError("Unsupported TAS decode method or writer")
        if self.name == "upstream-default" and (self.decode_method, self.writer) != (
            "cpu",
            "ffmpeg",
        ):
            raise CompetitorError(
                "TAS upstream-default requires --decode-method=cpu --writer=ffmpeg"
            )

    @property
    def max_compute_processes(self) -> int:
        return 1 if self.writer == "nelux" else 2

    def as_parameters(self) -> dict[str, str | bool]:
        return {
            "execution_profile": self.name,
            "decode_method": self.decode_method,
            "writer": self.writer,
            "cuda_graph": True,
        }


def add_execution_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose only TAS I/O choices, not VapourSynth scheduling fields."""
    parser.add_argument(
        "--execution-profile",
        choices=["upstream-default", "tuned"],
        default="upstream-default",
        help="I/O profile; tuned requires explicit decode method and writer",
    )
    parser.add_argument("--decode-method", choices=["cpu", "nvdec"], default=None)
    parser.add_argument("--writer", choices=["ffmpeg", "nelux"], default=None)


def resolve_execution_profile(args: argparse.Namespace) -> TasExecutionProfile:
    """Resolve the fixed upstream preset or an explicit tuned I/O configuration."""
    if args.execution_profile == "tuned":
        missing = [
            option
            for option, value in (
                ("--decode-method", args.decode_method),
                ("--writer", args.writer),
            )
            if value is None
        ]
        if missing:
            raise CompetitorError(f"TAS tuned requires explicit {', '.join(missing)}")
    return TasExecutionProfile(
        name=args.execution_profile,
        decode_method=args.decode_method or "cpu",
        writer=args.writer or "ffmpeg",
    )


def validate_declared_profile(implementation: dict[str, Any], profile: TasExecutionProfile) -> None:
    """Check executable upstream defaults against the pinned implementation metadata."""
    if profile.name == "tuned":
        return
    declared = implementation.get("execution_profiles", {}).get(profile.name)
    if not isinstance(declared, dict):
        raise CompetitorError(f"Implementation does not declare the {profile.name} profile")
    expected = profile.as_parameters()
    mismatches = [
        key
        for key in ("decode_method", "writer", "cuda_graph")
        if type(declared.get(key)) is not type(expected[key]) or declared.get(key) != expected[key]
    ]
    if mismatches:
        raise CompetitorError(
            "Execution profile metadata differs from the TAS runner preset: "
            + ", ".join(mismatches)
        )
