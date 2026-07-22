#!/usr/bin/env python3
"""Run or resume a rotated campaign and persist its actual execution order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.scripts.campaign import (
    EVENT_LOG_NAME,
    CampaignEvent,
    CampaignEventError,
    CampaignStep,
    append_event,
    campaign_steps,
    load_events,
    parse_timestamp,
    validate_event_prefix,
)


class CampaignRunError(RuntimeError):
    """Raised when campaign execution cannot preserve the benchmark contract."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _manifest_is_valid(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("status") == "valid"


def _verify_tracked_manifests(
    campaign_dir: Path,
    completed: list[CampaignEvent],
) -> None:
    tracked = {event.manifest for event in completed}
    for event in completed:
        path = campaign_dir / event.manifest
        if not _manifest_is_valid(path):
            raise CampaignRunError(f"Tracked campaign manifest is missing or invalid: {path}")

    discovered = {
        path.relative_to(campaign_dir).as_posix()
        for path in campaign_dir.glob("*/round-*/run-01/manifest.json")
    }
    untracked = sorted(discovered - tracked)
    if untracked:
        raise CampaignRunError(
            "Campaign contains manifests without completed execution events: "
            + ", ".join(untracked)
        )


def _wait_for_idle(previous: CampaignEvent | None, idle_seconds: float) -> float:
    if previous is None:
        return 0.0
    previous_finished = parse_timestamp(previous.finished_at_utc)
    elapsed = max(0.0, (_utc_now() - previous_finished).total_seconds())
    remaining = max(0.0, idle_seconds - elapsed)
    if remaining > 0:
        time.sleep(remaining)
    return max(0.0, (_utc_now() - previous_finished).total_seconds())


def _run_step(
    step: CampaignStep,
    *,
    attempt_index: int,
    previous: CampaignEvent | None,
    campaign_dir: Path,
    benchmarks_dir: Path,
    make_command: str,
    make_campaign_dir: str,
    idle_seconds: float,
    resume: bool,
) -> CampaignEvent:
    required_idle = 0.0 if previous is None else idle_seconds
    observed_idle = _wait_for_idle(previous, required_idle)
    started = _utc_now()
    started_clock = time.perf_counter()
    command = [
        make_command,
        "-C",
        str(benchmarks_dir),
        step.target,
        f"ROUND={step.round_index:02d}",
        f"CAMPAIGN_DIR={make_campaign_dir}",
        f"RESUME={int(resume)}",
    ]
    process = subprocess.run(command, check=False)
    finished = _utc_now()
    manifest_path = campaign_dir / step.manifest_path
    valid_manifest = _manifest_is_valid(manifest_path)
    completed = process.returncode == 0 and valid_manifest
    returncode = process.returncode if process.returncode != 0 else (0 if valid_manifest else 2)
    return CampaignEvent(
        schema_version=1,
        attempt_index=attempt_index,
        sequence_index=step.sequence_index,
        round_index=step.round_index,
        implementation=step.implementation,
        status="completed" if completed else "failed",
        returncode=returncode,
        required_idle_seconds=required_idle,
        observed_idle_seconds=observed_idle,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=time.perf_counter() - started_clock,
        manifest=step.manifest_path.as_posix(),
    )


def _run_until(
    rounds: int,
    *,
    events_path: Path,
    campaign_dir: Path,
    benchmarks_dir: Path,
    make_command: str,
    make_campaign_dir: str,
    idle_seconds: float,
    resume: bool,
) -> list[CampaignEvent]:
    events = load_events(events_path)
    completed = validate_event_prefix(events, idle_seconds=idle_seconds)
    _verify_tracked_manifests(campaign_dir, completed)
    expected = campaign_steps(rounds)
    for step in expected[len(completed) :]:
        previous = completed[-1] if completed else None
        event = _run_step(
            step,
            attempt_index=len(events) + 1,
            previous=previous,
            campaign_dir=campaign_dir,
            benchmarks_dir=benchmarks_dir,
            make_command=make_command,
            make_campaign_dir=make_campaign_dir,
            idle_seconds=idle_seconds,
            resume=resume,
        )
        append_event(events_path, event)
        events.append(event)
        if event.status != "completed":
            raise CampaignRunError(
                f"{step.implementation} round {step.round_index} failed; "
                f"see {campaign_dir / step.manifest_path.parent}"
            )
        completed.append(event)
    return completed


def _aggregate(
    *,
    benchmarks_dir: Path,
    make_command: str,
    make_campaign_dir: str,
    request_extra: bool,
) -> int:
    args = "--request-extra-exit-code" if request_extra else ""
    command = [
        make_command,
        "-C",
        str(benchmarks_dir),
        "aggregate-campaign",
        f"CAMPAIGN_DIR={make_campaign_dir}",
        f"ARGS={args}",
    ]
    return subprocess.run(command, check=False).returncode


def run_campaign(args: argparse.Namespace) -> int:
    campaign_dir = Path(args.campaign_dir).resolve()
    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    events_path = campaign_dir / EVENT_LOG_NAME
    if not args.resume and events_path.exists():
        raise CampaignRunError(
            f"Campaign event log already exists; use --resume: {events_path}"
        )

    completed = _run_until(
        3,
        events_path=events_path,
        campaign_dir=campaign_dir,
        benchmarks_dir=benchmarks_dir,
        make_command=args.make_command,
        make_campaign_dir=args.make_campaign_dir,
        idle_seconds=args.idle_seconds,
        resume=args.resume,
    )
    extras_started = len(completed) > len(campaign_steps(3))
    if extras_started:
        _run_until(
            5,
            events_path=events_path,
            campaign_dir=campaign_dir,
            benchmarks_dir=benchmarks_dir,
            make_command=args.make_command,
            make_campaign_dir=args.make_campaign_dir,
            idle_seconds=args.idle_seconds,
            resume=True,
        )
        return _aggregate(
            benchmarks_dir=benchmarks_dir,
            make_command=args.make_command,
            make_campaign_dir=args.make_campaign_dir,
            request_extra=False,
        )

    status = _aggregate(
        benchmarks_dir=benchmarks_dir,
        make_command=args.make_command,
        make_campaign_dir=args.make_campaign_dir,
        request_extra=True,
    )
    if status == 3:
        _run_until(
            5,
            events_path=events_path,
            campaign_dir=campaign_dir,
            benchmarks_dir=benchmarks_dir,
            make_command=args.make_command,
            make_campaign_dir=args.make_campaign_dir,
            idle_seconds=args.idle_seconds,
            resume=True,
        )
        return _aggregate(
            benchmarks_dir=benchmarks_dir,
            make_command=args.make_command,
            make_campaign_dir=args.make_campaign_dir,
            request_extra=False,
        )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a rotated benchmark campaign")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--make-campaign-dir", required=True)
    parser.add_argument("--benchmarks-dir", required=True)
    parser.add_argument("--idle-seconds", required=True, type=float)
    parser.add_argument("--make-command", default="make")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.idle_seconds < 0:
            raise CampaignRunError("--idle-seconds cannot be negative")
        returncode = run_campaign(args)
    except (CampaignEventError, CampaignRunError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
