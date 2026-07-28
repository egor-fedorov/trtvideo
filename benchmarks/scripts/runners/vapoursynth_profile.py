"""Execution profiles shared by VapourSynth benchmark runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

from benchmarks.scripts.contracts.benchmark import CompetitorError

Implementation = Literal["vstrt", "vsgan"]
Mode = Literal["upstream-default", "tuned"]
AutoOrInt = Literal["auto"] | int


@dataclass(frozen=True)
class VapourSynthExecutionProfile:
    """Resolved scheduling contract for one VapourSynth benchmark process."""

    mode: Mode
    requests: int | None
    num_streams: int
    vapoursynth_threads: int | None
    cuda_graph: bool

    def as_parameters(self) -> dict[str, str | int | bool]:
        return {
            "mode": self.mode,
            "vspipe_requests": self.requests if self.requests is not None else "auto",
            "num_streams": self.num_streams,
            "vapoursynth_threads": (
                self.vapoursynth_threads
                if self.vapoursynth_threads is not None
                else "auto"
            ),
            "cuda_graph": self.cuda_graph,
        }


_PRESETS: dict[tuple[Implementation, Mode], VapourSynthExecutionProfile] = {
    ("vstrt", "upstream-default"): VapourSynthExecutionProfile(
        mode="upstream-default",
        requests=None,
        num_streams=1,
        vapoursynth_threads=None,
        cuda_graph=False,
    ),
    ("vsgan", "upstream-default"): VapourSynthExecutionProfile(
        mode="upstream-default",
        requests=None,
        num_streams=4,
        vapoursynth_threads=4,
        cuda_graph=False,
    ),
}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _auto_or_positive_int(value: str) -> AutoOrInt:
    if value == "auto":
        return "auto"
    return _positive_int(value)


def add_execution_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common profile and scheduling options to a runner parser."""
    parser.add_argument(
        "--mode",
        choices=["upstream-default", "tuned"],
        default="upstream-default",
        help="Scheduling profile; tuned requires every scheduling option",
    )
    parser.add_argument(
        "--requests",
        type=_auto_or_positive_int,
        default=None,
        metavar="auto|N",
        help="Concurrent vspipe requests; auto omits --requests",
    )
    parser.add_argument(
        "--num-streams",
        type=_positive_int,
        default=None,
        help="Concurrent vstrt CUDA streams",
    )
    parser.add_argument(
        "--vs-threads",
        type=_auto_or_positive_int,
        default=None,
        metavar="auto|N",
        help="VapourSynth core threads; auto retains the runtime default",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or explicitly disable vstrt CUDA Graph",
    )


def _normalize_auto(value: AutoOrInt) -> int | None:
    return None if value == "auto" else value


def _format_auto(value: int | None) -> str:
    return "auto" if value is None else str(value)


def _resolve_preset(
    args: argparse.Namespace,
    implementation: Implementation,
    mode: Mode,
) -> VapourSynthExecutionProfile:
    profile = _PRESETS[(implementation, mode)]
    requested = (
        ("--requests", args.requests, profile.requests, _normalize_auto),
        ("--num-streams", args.num_streams, profile.num_streams, lambda value: value),
        (
            "--vs-threads",
            args.vs_threads,
            profile.vapoursynth_threads,
            _normalize_auto,
        ),
        ("--cuda-graph", args.cuda_graph, profile.cuda_graph, lambda value: value),
    )
    for option, override, expected, normalize in requested:
        if override is not None and normalize(override) != expected:
            expected_value = (
                _format_auto(expected)
                if option in {"--requests", "--vs-threads"}
                else str(expected).lower()
            )
            raise CompetitorError(
                f"{implementation} {mode} requires {option}={expected_value}"
            )
    return profile


def _resolve_tuned(
    args: argparse.Namespace,
    implementation: Implementation,
) -> VapourSynthExecutionProfile:
    required = {
        "--requests": args.requests,
        "--num-streams": args.num_streams,
        "--vs-threads": args.vs_threads,
        "--cuda-graph/--no-cuda-graph": args.cuda_graph,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        raise CompetitorError(
            f"{implementation} tuned requires explicit {', '.join(missing)}"
        )
    return VapourSynthExecutionProfile(
        mode="tuned",
        requests=_normalize_auto(args.requests),
        num_streams=args.num_streams,
        vapoursynth_threads=_normalize_auto(args.vs_threads),
        cuda_graph=args.cuda_graph,
    )


def resolve_execution_profile(
    args: argparse.Namespace,
    implementation: Implementation,
) -> VapourSynthExecutionProfile:
    """Resolve CLI overrides into a validated execution profile."""
    mode: Mode = args.mode
    if mode == "tuned":
        return _resolve_tuned(args, implementation)
    return _resolve_preset(args, implementation, mode)


def validate_declared_profile(
    implementation: dict[str, Any],
    profile: VapourSynthExecutionProfile,
) -> None:
    """Reject drift between executable presets and pinned source metadata."""
    if profile.mode == "tuned":
        return
    declared = implementation.get("execution_profiles", {}).get(profile.mode)
    if not isinstance(declared, dict):
        raise CompetitorError(
            f"Implementation does not declare the {profile.mode} profile"
        )
    actual = profile.as_parameters()
    expected = {
        "vspipe_requests": actual["vspipe_requests"],
        "num_streams": actual["num_streams"],
        "vapoursynth_threads": actual["vapoursynth_threads"],
        "cuda_graph": actual["cuda_graph"],
    }
    mismatches = [
        key for key, value in expected.items() if declared.get(key) != value
    ]
    if mismatches:
        raise CompetitorError(
            "Execution profile metadata differs from the runner preset: "
            + ", ".join(mismatches)
        )
