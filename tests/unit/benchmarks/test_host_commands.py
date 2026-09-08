from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "module",
    [
        "workflow.cli",
        "quality.preflight_tas",
        "tuning.workflow",
        "tuning.rank",
        "tuning.matrix",
        "campaign.run",
    ],
)
def test_host_command_imports_without_installed_packages(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"benchmarks.scripts.{module}", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("previous_path", [None, "/existing module path"])
def test_launcher_sets_checkout_path_for_host_python_and_child_commands(
    tmp_path: Path, previous_path: str | None
) -> None:
    interpreter = tmp_path / "host python"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$PYTHONPATH" "$PWD" "$@"\n')
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["HOST_PYTHON"] = str(interpreter)
    if previous_path is not None:
        env["PYTHONPATH"] = previous_path
    result = subprocess.run(
        [str(ROOT / "benchmarks/bin/run-benchmark.sh"), "tuned", "--resume"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected_path = str(ROOT / "src") + (f":{previous_path}" if previous_path else "")
    assert result.stdout.splitlines() == [
        expected_path,
        str(ROOT),
        "-m",
        "benchmarks.scripts.workflow.cli",
        "tuned",
        "--resume",
    ]
