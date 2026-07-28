"""CV-CUDA frame processing shared by production and quality capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trtvideo.video.nvcodec.surface import nv12_nhwc_view

if TYPE_CHECKING:
    from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime


@dataclass
class CvcudaFrameBuffers:
    """CV-CUDA buffers reused by every frame in one NVCodec job."""

    rgb_in_u8: Any
    rgb_in_float: Any
    rgb_out_float: Any
    rgb_out_u8: Any
    nv12_out: Any
    nv12_out_hwc: Any

    @classmethod
    def create(cls, runtime: CvcudaTensorRTRuntime) -> CvcudaFrameBuffers:
        cvcuda = runtime.cvcuda
        rgb_in_shape = (1, runtime.input_h, runtime.input_w, 3)
        rgb_out_shape = (1, runtime.output_h, runtime.output_w, 3)
        nv12_shape = (1, runtime.output_h * 3 // 2, runtime.output_w, 1)
        nv12_out = cvcuda.Tensor(nv12_shape, cvcuda.Type.U8, layout="NHWC")
        return cls(
            rgb_in_u8=cvcuda.Tensor(rgb_in_shape, cvcuda.Type.U8, layout="NHWC"),
            rgb_in_float=cvcuda.Tensor(
                rgb_in_shape,
                runtime.input_dtype,
                layout="NHWC",
            ),
            rgb_out_float=cvcuda.Tensor(
                rgb_out_shape,
                runtime.output_dtype,
                layout="NHWC",
            ),
            rgb_out_u8=cvcuda.Tensor(rgb_out_shape, cvcuda.Type.U8, layout="NHWC"),
            nv12_out=nv12_out,
            nv12_out_hwc=nv12_out.reshape(
                (runtime.output_h * 3 // 2, runtime.output_w, 1),
                layout="HWC",
            ),
        )


class NvcodecFrameProcessor:
    """Exact CV-CUDA/TensorRT frame path shared with model-space capture."""

    def __init__(
        self,
        runtime: CvcudaTensorRTRuntime,
        *,
        color_spec_name: str,
    ) -> None:
        self.runtime = runtime
        self.cvcuda = runtime.cvcuda
        self.color_spec = getattr(
            self.cvcuda.ColorSpec,
            color_spec_name.upper(),
        )
        self.buffers = CvcudaFrameBuffers.create(runtime)

    def wrap_nv12(self, raw_frame: Any) -> Any:
        """Wrap a pitch-padded decoder surface as visible NV12 without a copy."""
        source = self.cvcuda.as_tensor(raw_frame)
        view = nv12_nhwc_view(
            source,
            height=self.runtime.input_h,
            width=self.runtime.input_w,
        )
        return self.cvcuda.as_tensor(view, layout="NHWC")

    def preprocess(self, nv12_input: Any) -> Any:
        """Convert one NV12 frame into the bound normalized NCHW model input."""
        self.cvcuda.advcvtcolor_into(
            self.buffers.rgb_in_u8,
            nv12_input,
            self.cvcuda.ColorConversion.YUV2RGB_NV12,
            self.color_spec,
            stream=self.runtime.stream,
        )
        self.cvcuda.convertto_into(
            self.buffers.rgb_in_float,
            self.buffers.rgb_in_u8,
            scale=1.0 / 255.0,
            stream=self.runtime.stream,
        )
        self.cvcuda.reformat_into(
            self.runtime.gpu_input,
            self.buffers.rgb_in_float,
            stream=self.runtime.stream,
        )
        return self.runtime.gpu_input

    def infer(self) -> Any:
        """Enqueue TensorRT inference using the pre-bound model tensors."""
        return self.runtime.execute()

    def postprocess(self) -> Any:
        """Convert the bound NCHW model output into one encoder-ready NV12 frame."""
        self.cvcuda.reformat_into(
            self.buffers.rgb_out_float,
            self.runtime.gpu_output,
            stream=self.runtime.stream,
        )
        self.cvcuda.convertto_into(
            self.buffers.rgb_out_u8,
            self.buffers.rgb_out_float,
            scale=255.0,
            stream=self.runtime.stream,
        )
        self.cvcuda.advcvtcolor_into(
            self.buffers.nv12_out,
            self.buffers.rgb_out_u8,
            self.cvcuda.ColorConversion.RGB2YUV_NV12,
            self.color_spec,
            stream=self.runtime.stream,
        )
        return self.buffers.nv12_out_hwc
