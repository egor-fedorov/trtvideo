"""GPU colorspace conversion helpers."""

import cvcuda
import torch


def nv12_to_rgb_into(nv12, height, width, rgb_buf, nv12_buf=None):
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
    cvcuda.cvtcolor_into(rgb_cv, nv12_cv, cvcuda.ColorConversion.YUV2RGB_NV12)
    return rgb_buf


def nv12_to_rgb(nv12, height, width):
    """NV12 -> RGB on GPU via cvcuda with a newly allocated output buffer."""
    rgb_buf = torch.empty(height, width, 3, dtype=torch.uint8, device="cuda")
    return nv12_to_rgb_into(nv12, height, width, rgb_buf)


def rgb_to_nv12_into(rgb, nv12_buf):
    """RGB -> NV12 on GPU via cvcuda."""
    h, w = rgb.shape[:2]
    rgb_nhwc = rgb.unsqueeze(0)
    rgb_cv = cvcuda.as_tensor(rgb_nhwc, "NHWC")
    nv12_cv = cvcuda.as_tensor(nv12_buf.reshape(1, h * 3 // 2, w, 1), "NHWC")
    cvcuda.cvtcolor_into(nv12_cv, rgb_cv, cvcuda.ColorConversion.RGB2YUV_NV12)
    return nv12_buf


def rgb_to_nv12(rgb):
    """RGB -> NV12 on GPU via cvcuda with a newly allocated output buffer."""
    h, w = rgb.shape[:2]
    nv12_buf = torch.empty(h * 3 // 2, w, dtype=torch.uint8, device=rgb.device)
    return rgb_to_nv12_into(rgb, nv12_buf)
