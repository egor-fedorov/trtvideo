"""Execution contract for rotated benchmark campaigns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_UTC = timezone.utc  # noqa: UP017 - campaign orchestration supports host Python 3.10.

IMPLEMENTATIONS = {
    "ai-media": "ai-media-enhancer",
    "vstrt": "vs-mlrt",
    "vsgan": "VSGAN-tensorrt-docker",
}
ROUND_ORDERS = {
    1: ("ai-media", "vstrt", "vsgan"),
    2: ("vstrt", "vsgan", "ai-media"),
    3: ("vsgan", "ai-media", "vstrt"),
    4: ("vsgan", "vstrt", "ai-media"),
    5: ("ai-media", "vsgan", "vstrt"),
}
EVENT_LOG_NAME = "campaign.events.jsonl"
CONFIG_NAME = "campaign.config.json"
EXECUTION_PROFILES = ("parity", "upstream-default", "tuned")
IDLE_TOLERANCE_SECONDS = 0.05


class CampaignEventError(ValueError):
    """Raised when a campaign event log cannot prove the execution contract."""


@dataclass(frozen=True)
class CampaignConfig:
    """Immutable identity required to create or resume a campaign."""

    schema_version: int
    execution_profile: str
    vstrt_arguments: str
    vsgan_arguments: str

    @classmethod
    def create(
        cls,
        *,
        execution_profile: str,
        vstrt_arguments: str,
        vsgan_arguments: str,
    ) -> CampaignConfig:
        config = cls(
            schema_version=1,
            execution_profile=execution_profile,
            vstrt_arguments=vstrt_arguments.strip(),
            vsgan_arguments=vsgan_arguments.strip(),
        )
        config.validate()
        return config

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CampaignConfig:
        try:
            config = cls(
                schema_version=int(value["schema_version"]),
                execution_profile=str(value["execution_profile"]),
                vstrt_arguments=str(value["vstrt_arguments"]),
                vsgan_arguments=str(value["vsgan_arguments"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignEventError(f"Invalid campaign config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CampaignEventError(
                f"Unsupported campaign config schema: {self.schema_version}"
            )
        if self.execution_profile not in EXECUTION_PROFILES:
            raise CampaignEventError(
                f"Unknown campaign execution profile: {self.execution_profile}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignStep:
    """One product execution at a fixed position in the rotation."""

    sequence_index: int
    round_index: int
    implementation: str

    @property
    def target(self) -> str:
        return f"campaign-{self.implementation}"

    @property
    def manifest_path(self) -> Path:
        return (
            Path(self.implementation)
            / f"round-{self.round_index:02d}"
            / "run-01"
            / "manifest.json"
        )


@dataclass(frozen=True)
class CampaignEvent:
    """Durable evidence for one attempted campaign step."""

    schema_version: int
    attempt_index: int
    sequence_index: int
    round_index: int
    implementation: str
    status: str
    returncode: int
    required_idle_seconds: float
    observed_idle_seconds: float
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    manifest: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CampaignEvent:
        try:
            event = cls(
                schema_version=int(value["schema_version"]),
                attempt_index=int(value["attempt_index"]),
                sequence_index=int(value["sequence_index"]),
                round_index=int(value["round_index"]),
                implementation=str(value["implementation"]),
                status=str(value["status"]),
                returncode=int(value["returncode"]),
                required_idle_seconds=float(value["required_idle_seconds"]),
                observed_idle_seconds=float(value["observed_idle_seconds"]),
                started_at_utc=str(value["started_at_utc"]),
                finished_at_utc=str(value["finished_at_utc"]),
                duration_seconds=float(value["duration_seconds"]),
                manifest=str(value["manifest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignEventError(f"Invalid campaign event: {exc}") from exc
        event.validate_shape()
        return event

    def validate_shape(self) -> None:
        if self.schema_version != 1:
            raise CampaignEventError(
                f"Unsupported campaign event schema: {self.schema_version}"
            )
        if self.attempt_index <= 0 or self.sequence_index <= 0:
            raise CampaignEventError("Campaign event indexes must be positive")
        if self.status not in {"completed", "failed"}:
            raise CampaignEventError(f"Invalid campaign event status: {self.status}")
        if self.required_idle_seconds < 0 or self.observed_idle_seconds < 0:
            raise CampaignEventError("Campaign event idle durations cannot be negative")
        if self.duration_seconds < 0:
            raise CampaignEventError("Campaign event duration cannot be negative")
        parse_timestamp(self.started_at_utc)
        parse_timestamp(self.finished_at_utc)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def campaign_steps(rounds: int) -> tuple[CampaignStep, ...]:
    """Return the canonical execution sequence for three or five rounds."""
    if rounds not in {3, 5}:
        raise CampaignEventError(f"Campaign requires 3 or 5 rounds, got {rounds}")
    steps: list[CampaignStep] = []
    for round_index in range(1, rounds + 1):
        for implementation in ROUND_ORDERS[round_index]:
            steps.append(
                CampaignStep(
                    sequence_index=len(steps) + 1,
                    round_index=round_index,
                    implementation=implementation,
                )
            )
    return tuple(steps)


def parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CampaignEventError(f"Invalid campaign timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise CampaignEventError("Campaign timestamps must include a timezone")
    return timestamp.astimezone(_UTC)


def load_events(path: Path) -> list[CampaignEvent]:
    """Load an append-only JSONL event log."""
    if not path.exists():
        return []
    events = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CampaignEventError(f"Cannot read campaign event log {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignEventError(
                f"Invalid JSON in campaign event log line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CampaignEventError(
                f"Campaign event log line {line_number} is not an object"
            )
        events.append(CampaignEvent.from_dict(value))
    for expected_attempt, event in enumerate(events, start=1):
        if event.attempt_index != expected_attempt:
            raise CampaignEventError("Campaign attempt indexes are not contiguous")
    return events


def append_event(path: Path, event: CampaignEvent) -> None:
    """Append and flush one event after a campaign subprocess exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(event.as_dict(), file, sort_keys=True)
        file.write("\n")
        file.flush()


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load and validate one immutable campaign config."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignEventError(f"Cannot read campaign config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignEventError(f"Campaign config must be an object: {path}")
    return CampaignConfig.from_dict(value)


def write_campaign_config(path: Path, config: CampaignConfig) -> None:
    """Create a campaign config without replacing existing identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:
            json.dump(config.as_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
    except FileExistsError as exc:
        raise CampaignEventError(f"Campaign config already exists: {path}") from exc


def completed_events(events: list[CampaignEvent]) -> list[CampaignEvent]:
    return [event for event in events if event.status == "completed"]


def validate_event_prefix(
    events: list[CampaignEvent],
    *,
    idle_seconds: float,
) -> list[CampaignEvent]:
    """Validate every completed event as a canonical sequence prefix."""
    completed = completed_events(events)
    expected = campaign_steps(5)
    if len(completed) > len(expected):
        raise CampaignEventError("Campaign event log has too many completed steps")

    previous: CampaignEvent | None = None
    for position, event in enumerate(completed):
        step = expected[position]
        checks = {
            "sequence index": (event.sequence_index, step.sequence_index),
            "round index": (event.round_index, step.round_index),
            "implementation": (event.implementation, step.implementation),
            "manifest": (event.manifest, step.manifest_path.as_posix()),
            "return code": (event.returncode, 0),
        }
        for label, (actual, wanted) in checks.items():
            if actual != wanted:
                raise CampaignEventError(
                    f"Campaign event {position + 1} changed {label}: "
                    f"{actual!r} != {wanted!r}"
                )
        required_idle = 0.0 if previous is None else idle_seconds
        if event.required_idle_seconds != required_idle:
            raise CampaignEventError(
                f"Campaign event {position + 1} has an invalid idle requirement"
            )
        started = parse_timestamp(event.started_at_utc)
        finished = parse_timestamp(event.finished_at_utc)
        if finished < started:
            raise CampaignEventError(
                f"Campaign event {position + 1} finishes before it starts"
            )
        if previous is not None:
            previous_finished = parse_timestamp(previous.finished_at_utc)
            observed = (started - previous_finished).total_seconds()
            minimum = max(0.0, idle_seconds - IDLE_TOLERANCE_SECONDS)
            if observed < minimum or event.observed_idle_seconds < minimum:
                raise CampaignEventError(
                    f"Campaign event {position + 1} did not observe the required "
                    f"{idle_seconds:g}s idle interval"
                )
        previous = event
    return completed


def validate_complete_event_log(
    events: list[CampaignEvent],
    *,
    rounds: int,
    idle_seconds: float,
) -> list[CampaignEvent]:
    """Require exact event evidence for every result included in aggregation."""
    completed = validate_event_prefix(events, idle_seconds=idle_seconds)
    expected_count = len(campaign_steps(rounds))
    if len(completed) != expected_count:
        raise CampaignEventError(
            f"Campaign event log has {len(completed)} completed steps, "
            f"expected {expected_count}"
        )
    return completed
