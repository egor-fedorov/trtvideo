from __future__ import annotations

import ctypes
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _Frame:
    def __init__(self, *, width: int, height: int, stride: int) -> None:
        self.width = width
        self.height = height
        self.stride = stride
        self.planes = [(ctypes.c_ubyte * (stride * height))() for _ in range(3)]

    def copy(self) -> _Frame:
        return _Frame(width=self.width, height=self.height, stride=self.stride)

    def get_write_ptr(self, plane: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(ctypes.addressof(self.planes[plane]))

    def get_stride(self, _plane: int) -> int:
        return self.stride


class _Clip:
    def __init__(self, core: _Core, frame: _Frame) -> None:
        self.core = core
        self.frame = frame

    def set_output(self) -> None:
        self.core.output = self.frame


class _Std:
    def __init__(self, core: _Core) -> None:
        self.core = core

    def LoadPlugin(self, *, path: str) -> None:
        assert path == "/usr/local/lib/libvstrt.so"

    def BlankClip(
        self,
        *,
        width: int,
        height: int,
        format: object,
        length: int,
        fpsnum: int,
        fpsden: int,
    ) -> _Clip:
        assert format is self.core.rgbs
        assert (length, fpsnum, fpsden) == (1, 1, 1)
        row_bytes = width * 4
        return _Clip(
            self.core,
            _Frame(width=width, height=height, stride=row_bytes + 16),
        )

    def ModifyFrame(self, *, clip: _Clip, clips: _Clip, selector: object) -> _Clip:
        assert clip is clips
        return _Clip(self.core, selector(n=0, f=clip.frame))  # type: ignore[operator]

    def Limiter(self, clip: _Clip, *, min: float, max: float) -> _Clip:
        self.core.limiter_bounds.append((min, max))
        return clip


class _BestSource:
    def __init__(self, core: _Core) -> None:
        self.core = core

    def VideoSource(
        self,
        *,
        source: str,
        cachemode: int,
        showprogress: bool,
    ) -> _Clip:
        assert source == "/app/videos/input.mp4"
        assert cachemode == 0
        assert showprogress is False
        return _Clip(self.core, _Frame(width=2, height=2, stride=16))


class _Resize:
    def __init__(self, core: _Core) -> None:
        self.core = core

    def Bicubic(
        self,
        clip: _Clip,
        *,
        format: object,
        matrix_in_s: str,
        range_in_s: str,
    ) -> _Clip:
        assert format is self.core.rgbs
        assert matrix_in_s == "709"
        assert range_in_s == "limited"
        return clip


class _Core:
    def __init__(self) -> None:
        self.rgbs = object()
        self.std = _Std(self)
        self.bs = _BestSource(self)
        self.resize = _Resize(self)
        self.output: _Frame | None = None
        self.num_threads = 0
        self.limiter_bounds: list[tuple[float, float]] = []


@pytest.mark.parametrize(
    "script",
    [Path("benchmarks/vstrt/upscale.vpy"), Path("benchmarks/vsgan/upscale.vpy")],
)
def test_shared_input_scripts_copy_exact_rgb_planes_with_stride(
    script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 2
    height = 3
    values = np.arange(3 * width * height, dtype="<f4").reshape(3, height, width)
    tensor = tmp_path / "input.f32"
    values.tofile(tensor)

    core = _Core()
    vapoursynth = types.ModuleType("vapoursynth")
    vapoursynth.core = core  # type: ignore[attr-defined]
    vapoursynth.RGBS = core.rgbs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vapoursynth", vapoursynth)

    namespace = {
        "__name__": "__vapoursynth__",
        "model_input": str(tensor),
        "model_width": str(width),
        "model_height": str(height),
        "model_space_stage": "input",
    }
    exec(compile(script.read_bytes(), str(script), "exec"), namespace)

    assert core.output is not None
    assert core.limiter_bounds == []
    row_bytes = width * 4
    for plane in range(3):
        rows = []
        for row in range(height):
            start = row * core.output.stride
            rows.append(bytes(core.output.planes[plane][start : start + row_bytes]))
        actual = np.frombuffer(b"".join(rows), dtype="<f4").reshape(height, width)
        assert np.array_equal(actual, values[plane])


@pytest.mark.parametrize(
    "script",
    [Path("benchmarks/vstrt/upscale.vpy"), Path("benchmarks/vsgan/upscale.vpy")],
)
def test_production_scripts_clamp_decoded_rgbs_to_model_domain(script: Path) -> None:
    core = _Core()
    vapoursynth = types.ModuleType("vapoursynth")
    vapoursynth.core = core  # type: ignore[attr-defined]
    vapoursynth.RGBS = core.rgbs  # type: ignore[attr-defined]
    namespace = {
        "__name__": "__vapoursynth__",
        "source": "/app/videos/input.mp4",
        "model_space_stage": "input",
    }

    previous = sys.modules.get("vapoursynth")
    sys.modules["vapoursynth"] = vapoursynth
    try:
        exec(compile(script.read_bytes(), str(script), "exec"), namespace)
    finally:
        if previous is None:
            del sys.modules["vapoursynth"]
        else:
            sys.modules["vapoursynth"] = previous

    assert core.output is not None
    assert core.limiter_bounds == [(0.0, 1.0)]
