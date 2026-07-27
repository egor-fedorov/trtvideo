from __future__ import annotations

import pytest

from trtvideo.diagnostics.nvtx import NvtxAnnotator


def test_disabled_annotator_is_a_noop() -> None:
    calls: list[str] = []
    annotator = NvtxAnnotator(
        enabled=False,
        _push=lambda name: calls.append(name) or 0,
        _pop=lambda: calls.append("pop") or 0,
    )

    with annotator.range("stage"):
        calls.append("body")

    assert calls == ["body"]


def test_enabled_annotator_balances_range_on_error() -> None:
    calls: list[str] = []
    annotator = NvtxAnnotator(
        enabled=True,
        _push=lambda name: calls.append(f"push:{name}") or 0,
        _pop=lambda: calls.append("pop") or 0,
    )

    with pytest.raises(ValueError, match="failure"), annotator.range("stage"):
        calls.append("body")
        raise ValueError("failure")

    assert calls == ["push:stage", "body", "pop"]


def test_enabled_annotator_requires_callbacks() -> None:
    annotator = NvtxAnnotator(enabled=True)

    with pytest.raises(RuntimeError, match="without push/pop"), annotator.range("stage"):
        pass
