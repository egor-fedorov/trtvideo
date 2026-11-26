"""Publish a completed engine build without losing a previous build on ordinary I/O errors."""

import os
import shutil
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_destinations(outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
    seen = {path.resolve() for path in inputs}
    for path in outputs:
        resolved = path.resolve()
        overlaps = any(
            resolved.is_relative_to(other) or other.is_relative_to(resolved) for other in seen
        )
        if overlaps or path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Invalid or overlapping engine build destination: {path}")
        seen.add(resolved)


@contextmanager
def staged_build(outputs: Sequence[Path]) -> Generator[dict[Path, Path], None, None]:
    """Stage beside each destination. Publish in order, rolling back completed renames."""
    with ExitStack() as stack:
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path] = {}
        for dest in outputs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            directory = Path(
                stack.enter_context(TemporaryDirectory(prefix=".build-", dir=dest.parent))
            )
            staged[dest] = directory / "new"
        yield staged

        for dest, new in staged.items():
            if new.stat().st_size == 0:
                raise ValueError(f"Empty build artifact for {dest}")
            if dest.exists():
                backup = new.with_name("previous")
                try:
                    os.link(dest, backup)
                except OSError:
                    shutil.copy2(dest, backup)
                new.chmod(dest.stat().st_mode)
                backups[dest] = backup

        published: list[Path] = []
        try:
            for dest, new in staged.items():
                os.replace(new, dest)
                published.append(dest)
        except OSError:
            for dest in reversed(published):
                if dest in backups:
                    os.replace(backups[dest], dest)
                else:
                    dest.unlink()
            raise
