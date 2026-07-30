"""Shared policy and orchestration for repeated benchmark runs."""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

RunManifest = dict[str, Any]
RunExecutor = Callable[[int], RunManifest]
MetricReader = Callable[[RunManifest], float]
PowerLimitReader = Callable[[RunManifest], float | None]


@dataclass(frozen=True)
class SuitePolicy:
    """Rules governing repetition, noise handling and cooling intervals."""

    initial_runs: int
    extra_runs: int
    spread_threshold: float
    idle_seconds: float
    max_relative_spread: float | None = None

    def validate(self) -> None:
        if self.initial_runs <= 0:
            raise ValueError("initial_runs must be positive")
        if self.extra_runs < 0:
            raise ValueError("extra_runs cannot be negative")
        if not 0 <= self.spread_threshold < 1:
            raise ValueError("spread_threshold must be in the range [0, 1)")
        if (
            self.max_relative_spread is not None
            and not self.spread_threshold <= self.max_relative_spread < 1
        ):
            raise ValueError("max_relative_spread must be at least spread_threshold and below 1")
        if self.idle_seconds < 0:
            raise ValueError("idle_seconds cannot be negative")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> SuitePolicy:
        """Build a policy from the canonical benchmark parameter names."""
        policy = cls(
            initial_runs=int(parameters["initial_runs"]),
            extra_runs=int(parameters["extra_runs_on_spread"]),
            spread_threshold=float(parameters["spread_threshold"]),
            idle_seconds=float(parameters["idle_seconds"]),
            max_relative_spread=float(
                parameters.get(
                    "max_relative_spread",
                    parameters["spread_threshold"],
                )
            ),
        )
        policy.validate()
        return policy


@dataclass(frozen=True)
class SuiteResult:
    """Result of applying one repeat policy to a benchmark implementation."""

    status: str
    runs: tuple[RunManifest, ...]
    target_runs: int
    statistics: dict[str, Any]
    errors: tuple[str, ...]


def compute_suite_statistics(values: list[float]) -> dict[str, Any]:
    """Aggregate repeat measurements without hiding raw values."""
    if not values:
        return {
            "values_fps": [],
            "median_fps": None,
            "min_fps": None,
            "max_fps": None,
            "relative_spread": None,
        }
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "values_fps": values,
        "median_fps": median,
        "min_fps": minimum,
        "max_fps": maximum,
        "relative_spread": (maximum - minimum) / median if median > 0 else None,
    }


def should_extend_suite(values: list[float], threshold: float) -> bool:
    """Return whether initial measurements exceed the accepted noise threshold."""
    spread = compute_suite_statistics(values)["relative_spread"]
    return spread is not None and spread > threshold


def report_invalid_run(run: RunManifest, *, stream: TextIO | None = None) -> None:
    """Print manifest errors when a measured run invalidates its suite."""
    stream = stream or sys.stderr
    print(f"Benchmark run {run.get('run_index', '?')} invalid:", file=stream)
    errors = run.get("errors", [])
    if not errors:
        print("  - No detailed error was recorded", file=stream)
        return
    for error in errors:
        print(f"  - {error}", file=stream)


def report_publishability_errors(
    errors: list[str],
    *,
    acceptance_only: bool,
    stream: TextIO | None = None,
) -> None:
    """Report actionable standalone errors, not expected campaign-step state."""
    if not errors or acceptance_only:
        return
    stream = stream or sys.stderr
    print("Benchmark suite is not publishable:", file=stream)
    for error in errors:
        print(f"  - {error}", file=stream)


def canonical_suite_errors(
    parameters: Mapping[str, Any],
    benchmark: Mapping[str, Any] | None,
    *,
    include_warmup_frames: bool,
) -> list[str]:
    """Explain why suite parameters do not match the publication contract."""
    if benchmark is None:
        return ["No canonical workload benchmark contract was provided"]

    expected_keys = {
        "frames": "measured_frames",
        "initial_runs": "initial_runs",
        "extra_runs_on_spread": "extra_runs_on_spread",
        "spread_threshold": "spread_threshold",
        "max_relative_spread": "spread_threshold",
        "idle_seconds": "idle_seconds",
        "nvml_sample_interval_ms": "nvml_sample_interval_ms",
    }
    if include_warmup_frames:
        expected_keys["warmup_frames"] = "warmup_frames"

    errors = []
    for parameter_key, benchmark_key in expected_keys.items():
        actual = parameters.get(parameter_key)
        expected = benchmark.get(benchmark_key)
        if actual != expected:
            errors.append(
                f"{parameter_key} must match canonical {benchmark_key} ({actual!r} != {expected!r})"
            )
    return errors


def suite_publishability_errors(
    *,
    status: str,
    canonical_errors: list[str],
    runs: list[RunManifest] | tuple[RunManifest, ...],
    acceptance_only: bool = False,
) -> list[str]:
    """Collect suite-level reasons that prevent publishing a result."""
    errors = list(canonical_errors)
    if acceptance_only:
        errors.append("Individual suites are acceptance-only; use a rotated campaign result")
    if status != "valid":
        errors.append(f"Suite status is {status!r}, not 'valid'")
    for run in runs:
        run_index = run.get("run_index", "?")
        for error in run.get("reproducibility", {}).get("errors", []):
            errors.append(f"Run {run_index} reproducibility: {error}")
    return list(dict.fromkeys(errors))


class SuiteRunner:
    """Execute one implementation under a shared repeat policy."""

    def __init__(
        self,
        policy: SuitePolicy,
        *,
        label: str,
        frames: int,
        metric_reader: MetricReader,
        power_limit_reader: PowerLimitReader,
        sleep: Callable[[float], None] = time.sleep,
        stream: TextIO | None = None,
    ) -> None:
        policy.validate()
        self._policy = policy
        self._label = label
        self._frames = frames
        self._metric_reader = metric_reader
        self._power_limit_reader = power_limit_reader
        self._sleep = sleep
        self._stream = stream or sys.stderr

    def execute(self, run: RunExecutor) -> SuiteResult:
        """Execute initial runs, extend noisy suites and enforce shared invariants."""
        manifests: list[RunManifest] = []
        target_runs = self._policy.initial_runs
        run_index = 1
        while run_index <= target_runs:
            if run_index > 1 and self._policy.idle_seconds > 0:
                self._sleep(self._policy.idle_seconds)
            if self._policy.initial_runs == 1 and self._policy.extra_runs == 0:
                message = f"Benchmark: {self._label}, {self._frames} frames"
            else:
                message = (
                    f"Benchmark run {run_index}/{target_runs}: {self._label}, {self._frames} frames"
                )
            print(message, file=self._stream)
            manifest = run(run_index)
            manifests.append(manifest)
            if manifest.get("status") != "valid":
                report_invalid_run(manifest, stream=self._stream)
                break
            if run_index == self._policy.initial_runs and self._policy.extra_runs > 0:
                values = [self._metric_reader(item) for item in manifests]
                if should_extend_suite(values, self._policy.spread_threshold):
                    target_runs += self._policy.extra_runs
                    spread = compute_suite_statistics(values)["relative_spread"]
                    print(
                        f"Relative spread {spread:.2%} exceeds "
                        f"{self._policy.spread_threshold:.2%}; extending suite to "
                        f"{target_runs} runs",
                        file=self._stream,
                    )
            run_index += 1

        valid_runs = [item for item in manifests if item.get("status") == "valid"]
        values = [self._metric_reader(item) for item in valid_runs]
        statistics_report = compute_suite_statistics(values)
        errors = self._invariant_errors(valid_runs)
        all_valid = len(valid_runs) == len(manifests) == target_runs and not errors
        spread = statistics_report["relative_spread"]
        maximum_spread = (
            self._policy.max_relative_spread
            if self._policy.max_relative_spread is not None
            else self._policy.spread_threshold
        )
        stable = all_valid and spread is not None and spread <= maximum_spread
        status = "valid" if stable else ("unstable" if all_valid else "invalid")
        return SuiteResult(
            status=status,
            runs=tuple(manifests),
            target_runs=target_runs,
            statistics=statistics_report,
            errors=tuple(errors),
        )

    def _invariant_errors(self, runs: list[RunManifest]) -> list[str]:
        power_limits = {
            value for manifest in runs if (value := self._power_limit_reader(manifest)) is not None
        }
        if len(power_limits) > 1:
            return ["GPU power limit changed between measured runs"]
        return []
