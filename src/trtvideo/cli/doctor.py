"""Environment readiness CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trtvideo.diagnostics.doctor import DoctorReport, run_doctor


def build_parser() -> argparse.ArgumentParser:
    """Create the environment doctor parser."""
    parser = argparse.ArgumentParser(
        prog="trtvideo doctor",
        description="Check whether the static trtvideo runtime environment is ready",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index")
    parser.add_argument(
        "--disk-path",
        type=Path,
        default=Path.cwd(),
        help="Writable filesystem to inspect (default: current directory)",
    )
    return parser


def render_report(report: DoctorReport) -> str:
    """Render a stable human-readable readiness report."""
    width = max(len(check.component) for check in report.checks)
    lines = [
        f"[{('PASS' if check.passed else 'FAIL')}] {check.component:<{width}}  {check.detail}"
        for check in report.checks
    ]
    if report.ready:
        lines.append("Ready: static trtvideo runtime prerequisites are available.")
    else:
        failed = sum(not check.passed for check in report.checks)
        noun = "check" if failed == 1 else "checks"
        lines.append(f"Not ready: {failed} required {noun} failed.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run and print static environment readiness checks."""
    args = build_parser().parse_args(argv)
    report = run_doctor(gpu_id=args.gpu_id, disk_path=args.disk_path)
    print(render_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
