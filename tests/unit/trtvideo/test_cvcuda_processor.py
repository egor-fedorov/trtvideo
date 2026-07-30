from __future__ import annotations

from types import SimpleNamespace

import pytest

from trtvideo.video.nvcodec.processor import NvcodecFrameProcessor


class FakeTensor:
    _next_pointer = 1024

    def __init__(self, shape, dtype, layout, *, pointer=None, strides=None):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.pointer = pointer if pointer is not None else self._allocate_pointer()
        self.strides = strides

    @classmethod
    def _allocate_pointer(cls):
        pointer = cls._next_pointer
        cls._next_pointer += 4096
        return pointer

    def cuda(self):
        item_size = 1 if self.dtype == "u8" else 2
        return SimpleNamespace(
            __cuda_array_interface__={
                "version": 3,
                "shape": self.shape,
                "typestr": "|u1" if item_size == 1 else "<f2",
                "data": (self.pointer, False),
                "strides": self.strides,
            }
        )

    def reshape(self, shape, layout):
        return FakeTensor(shape, self.dtype, layout, pointer=self.pointer)


class FakeCvcuda:
    Type = SimpleNamespace(U8="u8")
    ColorSpec = SimpleNamespace(BT709="bt709")
    ColorConversion = SimpleNamespace(
        YUV2RGB_NV12="nv12-to-rgb",
        RGB2YUV_NV12="rgb-to-nv12",
    )

    def __init__(self):
        self.calls: list[tuple] = []

    def Tensor(self, shape, dtype, layout):
        return FakeTensor(shape, dtype, layout)

    def as_tensor(self, value, layout):
        interface = value.__cuda_array_interface__
        return FakeTensor(
            interface["shape"],
            "u8",
            layout,
            pointer=interface["data"][0],
            strides=interface["strides"],
        )

    def advcvtcolor_into(self, destination, source, conversion, spec, *, stream):
        self.calls.append(("advcvtcolor", destination, source, conversion, spec, stream))

    def convertto_into(self, destination, source, *, scale, offset=0.0, stream):
        self.calls.append(("convertto", destination, source, scale, offset, stream))

    def reformat_into(self, destination, source, *, stream):
        self.calls.append(("reformat", destination, source, stream))


class FakeRuntime:
    def __init__(self):
        self.cvcuda = FakeCvcuda()
        self.input_h = 2
        self.input_w = 4
        self.output_h = 4
        self.output_w = 8
        self.input_dtype = "f16"
        self.output_dtype = "f16"
        self.stream = object()
        self.gpu_input = FakeTensor((1, 3, 2, 4), "f16", "NCHW")
        self.gpu_output = FakeTensor((1, 3, 4, 8), "f16", "NCHW")
        self.executions = 0

    def execute(self):
        self.executions += 1
        return self.gpu_output


def test_frame_processor_shares_production_preprocess_and_inference() -> None:
    runtime = FakeRuntime()
    processor = NvcodecFrameProcessor(
        runtime,
        color_spec_name="bt709",
        limited_range=True,
    )
    nv12_input = FakeTensor((1, 3, 4, 1), "u8", "NHWC")

    model_input = processor.preprocess(nv12_input)
    model_output = processor.infer()
    encoder_input = processor.postprocess()

    assert model_input is runtime.gpu_input
    assert model_output is runtime.gpu_output
    assert runtime.executions == 1
    assert encoder_input.shape == (6, 8, 1)
    assert [call[0] for call in runtime.cvcuda.calls] == [
        "convertto",
        "convertto",
        "advcvtcolor",
        "convertto",
        "reformat",
        "reformat",
        "convertto",
        "advcvtcolor",
        "convertto",
        "convertto",
    ]
    y_to_full = runtime.cvcuda.calls[0]
    uv_to_full = runtime.cvcuda.calls[1]
    y_to_limited = runtime.cvcuda.calls[8]
    uv_to_limited = runtime.cvcuda.calls[9]
    assert y_to_full[3:5] == pytest.approx((255 / 219, -16 * 255 / 219))
    assert uv_to_full[3:5] == pytest.approx((255 / 224, 128 * (1 - 255 / 224)))
    assert y_to_limited[3:5] == pytest.approx((219 / 255, 16))
    assert uv_to_limited[3:5] == pytest.approx((224 / 255, 128 * (1 - 224 / 255)))


def test_frame_processor_keeps_full_range_nv12_without_range_conversion() -> None:
    runtime = FakeRuntime()
    processor = NvcodecFrameProcessor(
        runtime,
        color_spec_name="bt709",
        limited_range=False,
    )
    nv12_input = FakeTensor((1, 3, 4, 1), "u8", "NHWC")

    processor.preprocess(nv12_input)
    processor.postprocess()

    assert [call[0] for call in runtime.cvcuda.calls] == [
        "advcvtcolor",
        "convertto",
        "reformat",
        "reformat",
        "convertto",
        "advcvtcolor",
    ]
