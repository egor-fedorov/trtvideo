"""Opt-in NVTX ranges for external GPU timeline profilers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

NVTX_ENV = "AI_MEDIA_NVTX"


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

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"{NVTX_ENV}=1 requires PyTorch NVTX support") from exc

        return cls(
            enabled=True,
            _push=torch.cuda.nvtx.range_push,
            _pop=torch.cuda.nvtx.range_pop,
        )

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
