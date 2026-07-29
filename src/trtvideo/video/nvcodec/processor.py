"""CV-CUDA frame processing shared by production and quality capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trtvideo.video.nvcodec.surface import nv12_nhwc_view, nv12_plane_views

if TYPE_CHECKING:
    from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime

_CODE_VALUE_MAX = 255.0
_LIMITED_Y_MIN = 16.0
_LIMITED_Y_RANGE = 219.0
_LIMITED_CHROMA_CENTER = 128.0
_LIMITED_CHROMA_RANGE = 224.0


@dataclass
class Nv12FrameBuffer:
    """Owned NV12 tensor with reusable Y and interleaved UV plane views."""

    tensor: Any
    y: Any
    uv: Any
    hwc: Any

    @classmethod
    def create(
        cls,
        cvcuda: Any,
        *,
        height: int,
        width: int,
    ) -> Nv12FrameBuffer:
        tensor = cvcuda.Tensor(
            (1, height * 3 // 2, width, 1),
            cvcuda.Type.U8,
            layout="NHWC",
        )
        y_view, uv_view = nv12_plane_views(
            tensor,
            height=height,
            width=width,
        )
        return cls(
            tensor=tensor,
            y=cvcuda.as_tensor(y_view, layout="NHWC"),
            uv=cvcuda.as_tensor(uv_view, layout="NHWC"),
            hwc=tensor.reshape(
                (height * 3 // 2, width, 1),
                layout="HWC",
            ),
        )


@dataclass
class CvcudaFrameBuffers:
    """CV-CUDA buffers reused by every frame in one NVCodec job."""

    rgb_in_u8: Any
    rgb_in_float: Any
    rgb_out_float: Any
    rgb_out_u8: Any
    nv12_in_full: Nv12FrameBuffer
    nv12_out_full: Nv12FrameBuffer
    nv12_out_limited: Nv12FrameBuffer

    @classmethod
    def create(cls, runtime: CvcudaTensorRTRuntime) -> CvcudaFrameBuffers:
        cvcuda = runtime.cvcuda
        rgb_in_shape = (1, runtime.input_h, runtime.input_w, 3)
        rgb_out_shape = (1, runtime.output_h, runtime.output_w, 3)
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
            nv12_in_full=Nv12FrameBuffer.create(
                cvcuda,
                height=runtime.input_h,
                width=runtime.input_w,
            ),
            nv12_out_full=Nv12FrameBuffer.create(
                cvcuda,
                height=runtime.output_h,
                width=runtime.output_w,
            ),
            nv12_out_limited=Nv12FrameBuffer.create(
                cvcuda,
                height=runtime.output_h,
                width=runtime.output_w,
            ),
        )


class NvcodecFrameProcessor:
    """Exact CV-CUDA/TensorRT frame path shared with model-space capture."""

    def __init__(
        self,
        runtime: CvcudaTensorRTRuntime,
        *,
        color_spec_name: str,
        limited_range: bool,
    ) -> None:
        self.runtime = runtime
        self.cvcuda = runtime.cvcuda
        self.limited_range = limited_range
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
        color_input = nv12_input
        if self.limited_range:
            source_y_view, source_uv_view = nv12_plane_views(
                nv12_input,
                height=self.runtime.input_h,
                width=self.runtime.input_w,
            )
            source_y = self.cvcuda.as_tensor(source_y_view, layout="NHWC")
            source_uv = self.cvcuda.as_tensor(source_uv_view, layout="NHWC")
            self.cvcuda.convertto_into(
                self.buffers.nv12_in_full.y,
                source_y,
                scale=_CODE_VALUE_MAX / _LIMITED_Y_RANGE,
                offset=-_LIMITED_Y_MIN * _CODE_VALUE_MAX / _LIMITED_Y_RANGE,
                stream=self.runtime.stream,
            )
            self.cvcuda.convertto_into(
                self.buffers.nv12_in_full.uv,
                source_uv,
                scale=_CODE_VALUE_MAX / _LIMITED_CHROMA_RANGE,
                offset=(
                    _LIMITED_CHROMA_CENTER
                    * (1.0 - _CODE_VALUE_MAX / _LIMITED_CHROMA_RANGE)
                ),
                stream=self.runtime.stream,
            )
            color_input = self.buffers.nv12_in_full.tensor
        self.cvcuda.advcvtcolor_into(
            self.buffers.rgb_in_u8,
            color_input,
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
            self.buffers.nv12_out_full.tensor,
            self.buffers.rgb_out_u8,
            self.cvcuda.ColorConversion.RGB2YUV_NV12,
            self.color_spec,
            stream=self.runtime.stream,
        )
        if not self.limited_range:
            return self.buffers.nv12_out_full.hwc
        self.cvcuda.convertto_into(
            self.buffers.nv12_out_limited.y,
            self.buffers.nv12_out_full.y,
            scale=_LIMITED_Y_RANGE / _CODE_VALUE_MAX,
            offset=_LIMITED_Y_MIN,
            stream=self.runtime.stream,
        )
        self.cvcuda.convertto_into(
            self.buffers.nv12_out_limited.uv,
            self.buffers.nv12_out_full.uv,
            scale=_LIMITED_CHROMA_RANGE / _CODE_VALUE_MAX,
            offset=(
                _LIMITED_CHROMA_CENTER
                * (1.0 - _LIMITED_CHROMA_RANGE / _CODE_VALUE_MAX)
            ),
            stream=self.runtime.stream,
        )
        return self.buffers.nv12_out_limited.hwc
