from __future__ import annotations

import numpy as np
import pytest

from trtvideo.video.cuda_array import nv12_nhwc_view


class FakeCudaBuffer:
    def __init__(self, interface):
        self.__cuda_array_interface__ = interface


class FakeTensor:
    def __init__(self, interface):
        self.buffer = FakeCudaBuffer(interface)

    def cuda(self):
        return self.buffer


def test_nv12_view_crops_pitch_padding_without_copying() -> None:
    source = FakeTensor(
        {
            "version": 3,
            "shape": (1080, 1344),
            "typestr": np.dtype(np.uint8).str,
            "data": (123456, False),
            "strides": (1344, 1),
        }
    )

    view = nv12_nhwc_view(source, height=720, width=1280)

    assert view.__cuda_array_interface__ == {
        "version": 3,
        "shape": (1, 1080, 1280, 1),
        "typestr": "|u1",
        "data": (123456, False),
        "strides": (1_451_520, 1344, 1, 1),
    }
    assert view.owner[0] is source


def test_nv12_view_rejects_surface_smaller_than_visible_frame() -> None:
    source = FakeTensor(
        {
            "version": 3,
            "shape": (100, 100),
            "typestr": np.dtype(np.uint8).str,
            "data": (123456, False),
            "strides": None,
        }
    )

    with pytest.raises(ValueError, match="smaller than the visible frame"):
        nv12_nhwc_view(source, height=720, width=1280)


def test_nv12_view_accepts_single_channel_hwc_surface() -> None:
    source = FakeTensor(
        {
            "version": 3,
            "shape": (1080, 1344, 1),
            "typestr": np.dtype(np.uint8).str,
            "data": (123456, False),
            "strides": (1344, 1, 1),
        }
    )

    view = nv12_nhwc_view(source, height=720, width=1280)

    assert view.__cuda_array_interface__["shape"] == (1, 1080, 1280, 1)
    assert view.__cuda_array_interface__["strides"] == (1_451_520, 1344, 1, 1)
