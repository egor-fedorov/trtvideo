from __future__ import annotations

from types import SimpleNamespace

from trtvideo.video.cvcuda import NvcodecFrameProcessor


class FakeTensor:
    def __init__(self, shape, dtype, layout):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def reshape(self, shape, layout):
        return FakeTensor(shape, self.dtype, layout)


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

    def advcvtcolor_into(self, destination, source, conversion, spec, *, stream):
        self.calls.append(("advcvtcolor", destination, source, conversion, spec, stream))

    def convertto_into(self, destination, source, *, scale, stream):
        self.calls.append(("convertto", destination, source, scale, stream))

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
    processor = NvcodecFrameProcessor(runtime, color_spec_name="bt709")
    nv12_input = FakeTensor((1, 3, 4, 1), "u8", "NHWC")

    model_input = processor.preprocess(nv12_input)
    model_output = processor.infer()
    encoder_input = processor.postprocess()

    assert model_input is runtime.gpu_input
    assert model_output is runtime.gpu_output
    assert runtime.executions == 1
    assert encoder_input.shape == (6, 8, 1)
    assert [call[0] for call in runtime.cvcuda.calls] == [
        "advcvtcolor",
        "convertto",
        "reformat",
        "reformat",
        "convertto",
        "advcvtcolor",
    ]
