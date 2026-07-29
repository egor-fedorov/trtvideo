"""Zero-copy CUDA Array Interface views for NVDEC surfaces."""

from __future__ import annotations

from typing import Any


class CudaArrayView:
    """Keep a CUDA buffer owner alive while exposing a different tensor view."""

    def __init__(self, interface: dict[str, Any], owner: Any):
        self.__cuda_array_interface__ = interface
        self.owner = owner


def nv12_nhwc_view(source: Any, *, height: int, width: int) -> CudaArrayView:
    """Expose a possibly pitch-padded NV12 surface as a cropped NHWC tensor."""
    cuda_buffer = source.cuda()
    interface = cuda_buffer.__cuda_array_interface__
    shape = tuple(int(value) for value in interface["shape"])
    typestr = str(interface["typestr"])
    expected_rows = height * 3 // 2

    if not typestr.endswith("u1"):
        raise ValueError(f"Expected uint8 NV12 surface, got {typestr}")
    if len(shape) == 2:
        source_rows, source_width = shape
        channels = 1
    elif len(shape) == 3 and shape[2] == 1:
        source_rows, source_width, channels = shape
    else:
        raise ValueError(f"Expected HW or HWC NV12 surface, got shape={shape}")
    if source_rows < expected_rows or source_width < width:
        raise ValueError(
            "Decoded NV12 surface is smaller than the visible frame: "
            f"surface={shape}, frame={width}x{height}"
        )

    strides = interface.get("strides")
    if strides is None:
        row_stride = source_width * channels
        pixel_stride = channels
        channel_stride = 1
    elif len(shape) == 2:
        row_stride, pixel_stride = (int(value) for value in strides)
        channel_stride = 1
    else:
        row_stride, pixel_stride, channel_stride = (int(value) for value in strides)

    return CudaArrayView(
        {
            "version": 3,
            "shape": (1, expected_rows, width, 1),
            "typestr": typestr,
            "data": (int(interface["data"][0]), False),
            "strides": (
                row_stride * expected_rows,
                row_stride,
                pixel_stride,
                channel_stride,
            ),
        },
        owner=(source, cuda_buffer),
    )


def nv12_plane_views(
    source: Any,
    *,
    height: int,
    width: int,
) -> tuple[CudaArrayView, CudaArrayView]:
    """Expose the Y and interleaved UV rows of an NHWC NV12 tensor."""
    cuda_buffer = source.cuda()
    interface = cuda_buffer.__cuda_array_interface__
    shape = tuple(int(value) for value in interface["shape"])
    expected_shape = (1, height * 3 // 2, width, 1)
    if shape != expected_shape:
        raise ValueError(f"Expected NV12 NHWC shape {expected_shape}, got {shape}")

    typestr = str(interface["typestr"])
    if not typestr.endswith("u1"):
        raise ValueError(f"Expected uint8 NV12 tensor, got {typestr}")

    strides = interface.get("strides")
    if strides is None:
        sample_stride = height * 3 // 2 * width
        row_stride = width
        pixel_stride = 1
        channel_stride = 1
    else:
        sample_stride, row_stride, pixel_stride, channel_stride = (
            int(value) for value in strides
        )

    pointer = int(interface["data"][0])
    owner = (source, cuda_buffer)

    def plane(rows: int, offset: int) -> CudaArrayView:
        return CudaArrayView(
            {
                "version": 3,
                "shape": (1, rows, width, 1),
                "typestr": typestr,
                "data": (pointer + offset, False),
                "strides": (
                    sample_stride,
                    row_stride,
                    pixel_stride,
                    channel_stride,
                ),
            },
            owner=owner,
        )

    return plane(height, 0), plane(height // 2, height * row_stride)
