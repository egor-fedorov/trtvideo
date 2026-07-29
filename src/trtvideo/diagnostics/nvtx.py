"""Opt-in NVTX ranges for external GPU timeline profilers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

NVTX_ENV = "TRTVIDEO_NVTX"


def _load_nvtx_callbacks() -> tuple[Callable[[str], int], Callable[[], int]]:
    try:
        import nvtx
    except ImportError as exc:
        raise RuntimeError(
            f"{NVTX_ENV}=1 requires the optional benchmark NVTX binding"
        ) from exc

    def push(name: str) -> int:
        nvtx.push_range(name)
        return 0

    def pop() -> int:
        nvtx.pop_range()
        return 0

    return push, pop


@dataclass(frozen=True)
class NvtxAnnotator:
    """Push NVTX ranges only when explicitly enabled for diagnostics."""

    enabled: bool
    _push: Callable[[str], int] | None = None
    _pop: Callable[[], int] | None = None

    @classmethod
    def from_environment(cls) -> NvtxAnnotator:
        """Create a lazy NVTX provider without affecting ordinary startup."""
        if os.environ.get(NVTX_ENV) != "1":
            return cls(enabled=False)

        push, pop = _load_nvtx_callbacks()
        return cls(enabled=True, _push=push, _pop=pop)

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        """Annotate one range or behave as a no-op when diagnostics are disabled."""
        if not self.enabled:
            yield
            return
        if self._push is None or self._pop is None:
            raise RuntimeError("NVTX annotator is enabled without push/pop callbacks")

        self._push(name)
        try:
            yield
        finally:
            self._pop()
