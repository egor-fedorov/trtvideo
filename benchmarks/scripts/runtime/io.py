"""JSON output helpers shared by benchmark commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.scripts.runtime.environment import write_json


def write_json_target(value: dict[str, Any], target: str | None) -> None:
    """Write a plan to a file or stdout; a missing target defaults to stdout."""
    if target is None or target == "-":
        json.dump(value, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    write_json(Path(target), value)


def write_summary_target(path: str | None, summary: dict[str, Any]) -> None:
    """Optionally mirror a canonical suite summary to a file or stdout."""
    if path is None:
        return
    if path == "-":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    write_json(Path(path), summary)
