"""Atomic output-container publication."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Literal

from trtvideo.video.output.preservation import (
    MediaPreservationError,
    preflight_output_container,
)


class AtomicOutputTransaction:
    """Preflight and atomically expose one completed output container."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        *,
        preserve_chapters: bool,
    ) -> None:
        self._input_path = input_path
        self._output_path = output_path
        self._preserve_chapters = preserve_chapters
        self._temporary_path: Path | None = None

    def __enter__(self) -> Path:
        preflight_output_container(
            self._input_path,
            self._output_path,
            preserve_chapters=self._preserve_chapters,
        )
        self._temporary_path = create_staging_output(self._output_path)
        return self._temporary_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_value, traceback
        temporary_path = self._temporary_path
        self._temporary_path = None
        if temporary_path is None:
            return False

        try:
            if exc_type is None:
                commit_atomic_output(temporary_path, self._output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return False


def create_staging_output(output_path: str) -> Path:
    """Reserve a same-directory temporary path suitable for atomic replacement."""
    output = Path(output_path)
    parent = output.parent
    if not parent.is_dir():
        raise MediaPreservationError(f"output directory does not exist: {parent}")
    if not output.suffix:
        raise MediaPreservationError(
            "output path must have a container extension such as .mkv or .mp4"
        )

    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=f".partial{output.suffix}",
        dir=parent,
    )
    os.close(fd)
    return Path(temporary)


def commit_atomic_output(temporary_path: Path, output_path: str) -> None:
    """Atomically expose a completed output without leaving partial media behind."""
    os.replace(temporary_path, output_path)
