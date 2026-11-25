"""CV-CUDA frame processing shared by production and quality capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from trtvideo.video.nvcodec.surface import nv12_nhwc_view, nv12_plane_views

if TYPE_CHECKING:
    import cvcuda
    from PyNvVideoCodec import DecodedFrame

    from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime

_CODE_VALUE_MAX = 255.0
_LIMITED_Y_MIN = 16.0
_LIMITED_Y_RANGE = 219.0
_LIMITED_CHROMA_CENTER = 128.0
_LIMITED_CHROMA_RANGE = 224.0


@dataclass
class Nv12FrameBuffer:
    """Owned NV12 tensor with reusable Y and interleaved UV plane views."""

    tensor: cvcuda.Tensor
    y: cvcuda.Tensor
    uv: cvcuda.Tensor
    hwc: cvcuda.Tensor

    @classmethod
    def create(
        cls,
        *,
        height: int,
        width: int,
    ) -> Nv12FrameBuffer:
        import cvcuda

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

    rgb_in_u8: cvcuda.Tensor
    rgb_in_float: cvcuda.Tensor
    rgb_out_float: cvcuda.Tensor
    rgb_out_u8: cvcuda.Tensor
    nv12_in_full: Nv12FrameBuffer
    nv12_out_full: Nv12FrameBuffer
    nv12_out_limited: Nv12FrameBuffer

    @classmethod
    def create(cls, runtime: CvcudaTensorRTRuntime) -> CvcudaFrameBuffers:
        import cvcuda

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
                height=runtime.input_h,
                width=runtime.input_w,
            ),
            nv12_out_full=Nv12FrameBuffer.create(
                height=runtime.output_h,
                width=runtime.output_w,
            ),
            nv12_out_limited=Nv12FrameBuffer.create(
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
        import cvcuda

        self.runtime = runtime
        # Retain callable signatures without storing an untyped ModuleType.
        self._as_tensor = cvcuda.as_tensor
        self._convertto_into = cvcuda.convertto_into
        self._reformat_into = cvcuda.reformat_into
        self._advcvtcolor_into = cvcuda.advcvtcolor_into
        self._nv12_to_rgb = cvcuda.ColorConversion.YUV2RGB_NV12
        self._rgb_to_nv12 = cvcuda.ColorConversion.RGB2YUV_NV12
        self.limited_range = limited_range
        self.color_spec = cast(
            "cvcuda.ColorSpec",
            getattr(cvcuda.ColorSpec, color_spec_name.upper()),
        )
        self.buffers = CvcudaFrameBuffers.create(runtime)

    def wrap_nv12(self, raw_frame: DecodedFrame) -> cvcuda.Tensor:
        """Wrap a pitch-padded decoder surface as visible NV12 without a copy."""
        source = self._as_tensor(raw_frame)
        view = nv12_nhwc_view(
            source,
            height=self.runtime.input_h,
            width=self.runtime.input_w,
        )
        return self._as_tensor(view, layout="NHWC")

    def preprocess(self, nv12_input: cvcuda.Tensor) -> cvcuda.Tensor:
        """Convert one NV12 frame into the bound normalized NCHW model input."""
        color_input = nv12_input
        if self.limited_range:
            source_y_view, source_uv_view = nv12_plane_views(
                nv12_input,
                height=self.runtime.input_h,
                width=self.runtime.input_w,
            )
            source_y = self._as_tensor(source_y_view, layout="NHWC")
            source_uv = self._as_tensor(source_uv_view, layout="NHWC")
            self._convertto_into(
                self.buffers.nv12_in_full.y,
                source_y,
                scale=_CODE_VALUE_MAX / _LIMITED_Y_RANGE,
                offset=-_LIMITED_Y_MIN * _CODE_VALUE_MAX / _LIMITED_Y_RANGE,
                stream=self.runtime.stream,
            )
            self._convertto_into(
                self.buffers.nv12_in_full.uv,
                source_uv,
                scale=_CODE_VALUE_MAX / _LIMITED_CHROMA_RANGE,
                offset=(_LIMITED_CHROMA_CENTER * (1.0 - _CODE_VALUE_MAX / _LIMITED_CHROMA_RANGE)),
                stream=self.runtime.stream,
            )
            color_input = self.buffers.nv12_in_full.tensor
        self._advcvtcolor_into(
            self.buffers.rgb_in_u8,
            color_input,
            self._nv12_to_rgb,
            self.color_spec,
            stream=self.runtime.stream,
        )
        self._convertto_into(
            self.buffers.rgb_in_float,
            self.buffers.rgb_in_u8,
            scale=1.0 / 255.0,
            stream=self.runtime.stream,
        )
        self._reformat_into(
            self.runtime.gpu_input,
            self.buffers.rgb_in_float,
            stream=self.runtime.stream,
        )
        return self.runtime.gpu_input

    def infer(self) -> cvcuda.Tensor:
        """Enqueue TensorRT inference using the pre-bound model tensors."""
        return self.runtime.execute()

    def postprocess(self) -> cvcuda.Tensor:
        """Convert the bound NCHW model output into one encoder-ready NV12 frame."""
        self._reformat_into(
            self.buffers.rgb_out_float,
            self.runtime.gpu_output,
            stream=self.runtime.stream,
        )
        self._convertto_into(
            self.buffers.rgb_out_u8,
            self.buffers.rgb_out_float,
            scale=255.0,
            stream=self.runtime.stream,
        )
        self._advcvtcolor_into(
            self.buffers.nv12_out_full.tensor,
            self.buffers.rgb_out_u8,
            self._rgb_to_nv12,
            self.color_spec,
            stream=self.runtime.stream,
        )
        if not self.limited_range:
            return self.buffers.nv12_out_full.hwc
        self._convertto_into(
            self.buffers.nv12_out_limited.y,
            self.buffers.nv12_out_full.y,
            scale=_LIMITED_Y_RANGE / _CODE_VALUE_MAX,
            offset=_LIMITED_Y_MIN,
            stream=self.runtime.stream,
        )
        self._convertto_into(
            self.buffers.nv12_out_limited.uv,
            self.buffers.nv12_out_full.uv,
            scale=_LIMITED_CHROMA_RANGE / _CODE_VALUE_MAX,
            offset=(_LIMITED_CHROMA_CENTER * (1.0 - _LIMITED_CHROMA_RANGE / _CODE_VALUE_MAX)),
            stream=self.runtime.stream,
        )
        return self.buffers.nv12_out_limited.hwc
