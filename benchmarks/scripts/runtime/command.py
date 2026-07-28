"""Shell-free command specifications shared by benchmark adapters."""

from __future__ import annotations

import shlex

CommandSpec = list[list[str]]


def command_spec(*commands: list[str]) -> CommandSpec:
    """Create an argv-only command or pipeline specification."""
    if not commands or any(not command for command in commands):
        raise ValueError("Command specification cannot be empty")
    return [list(command) for command in commands]


def display_command(spec: CommandSpec) -> str:
    """Render a command specification for logs and dry-run plans."""
    return " | ".join(shlex.join(command) for command in spec)
