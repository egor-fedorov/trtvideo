"""GPU colorspace conversion helpers."""

from typing import Any

import cvcuda
import torch

try:
    import nvcv
except ImportError:
    nvcv = None


def _resolve_color_spec(color_spec: str) -> Any | None:
    if nvcv is None:
        return None
    return getattr(nvcv.ColorSpec, color_spec.upper(), None)


def _advcvtcolor_into(dst, src, code, color_spec: str) -> None:
    spec = _resolve_color_spec(color_spec)
    if spec is None or not hasattr(cvcuda, "advcvtcolor_into"):
        cvcuda.cvtcolor_into(dst, src, code)
        return
    cvcuda.advcvtcolor_into(dst, src, code, spec)


def nv12_to_rgb_into(nv12, height, width, rgb_buf, nv12_buf=None, color_spec: str = "bt709"):
    """NV12 -> RGB on GPU via cvcuda."""
    if nv12.ndim == 2 and nv12.shape[1] != width:
        nv12 = nv12[:, :width]
    if not nv12.is_contiguous():
        if nv12_buf is None:
            nv12 = nv12.contiguous()
        else:
            nv12_buf.copy_(nv12)
            nv12 = nv12_buf
    nv12_nhwc = nv12.reshape(1, height * 3 // 2, width, 1)
    nv12_cv = cvcuda.as_tensor(nv12_nhwc, "NHWC")
    rgb_cv = cvcuda.as_tensor(rgb_buf.reshape(1, height, width, 3), "NHWC")
    _advcvtcolor_into(rgb_cv, nv12_cv, cvcuda.ColorConversion.YUV2RGB_NV12, color_spec)
    return rgb_buf


def nv12_to_rgb(nv12, height, width, color_spec: str = "bt709"):
    """NV12 -> RGB on GPU via cvcuda with a newly allocated output buffer."""
    rgb_buf = torch.empty(height, width, 3, dtype=torch.uint8, device="cuda")
    return nv12_to_rgb_into(nv12, height, width, rgb_buf, color_spec=color_spec)


def rgb_to_nv12_into(rgb, nv12_buf, color_spec: str = "bt709"):
    """RGB -> NV12 on GPU via cvcuda."""
    h, w = rgb.shape[:2]
    rgb_nhwc = rgb.unsqueeze(0)
    rgb_cv = cvcuda.as_tensor(rgb_nhwc, "NHWC")
    nv12_cv = cvcuda.as_tensor(nv12_buf.reshape(1, h * 3 // 2, w, 1), "NHWC")
    _advcvtcolor_into(nv12_cv, rgb_cv, cvcuda.ColorConversion.RGB2YUV_NV12, color_spec)
    return nv12_buf


def rgb_to_nv12(rgb, color_spec: str = "bt709"):
    """RGB -> NV12 on GPU via cvcuda with a newly allocated output buffer."""
    h, w = rgb.shape[:2]
    nv12_buf = torch.empty(h * 3 // 2, w, dtype=torch.uint8, device=rgb.device)
    return rgb_to_nv12_into(rgb, nv12_buf, color_spec=color_spec)
