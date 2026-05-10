"""Minimal model contracts for single-frame upscale tasks."""

from dataclasses import dataclass
from typing import Literal

TensorLayout = Literal["nchw"]
TensorDType = Literal["fp32"]
PixelFormat = Literal["rgb"]
TensorRange = Literal["0_1"]
TaskType = Literal["upscale"]


@dataclass(frozen=True)
class TensorSpec:
    """Tensor contract used by runtime and pipeline validation."""

    name: str
    layout: TensorLayout
    dtype: TensorDType
    shape: tuple[int, ...]
    pixel_format: PixelFormat
    value_range: TensorRange


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model contract for current SPAN/RealESRGAN-like upscale models."""

    name: str
    task: TaskType
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    scale: int
    temporal: bool = False
    preprocess: str = "uint8_to_float_0_1"
    postprocess: str = "float_0_1_to_uint8"


def make_upscale_model_spec(
    *,
    name: str,
    input_name: str,
    output_name: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> ModelSpec:
    """Create and validate a minimal single-frame upscale model spec."""
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ValueError(
            f"Upscale models must use NCHW rank-4 tensors, got {input_shape} -> {output_shape}."
        )

    in_n, in_c, in_h, in_w = input_shape
    out_n, out_c, out_h, out_w = output_shape

    if any(dim <= 0 for dim in (*input_shape, *output_shape)):
        raise ValueError(
            f"Static positive tensor shapes are required, got {input_shape} -> {output_shape}."
        )
    if in_n != 1 or out_n != 1:
        raise ValueError(f"Only batch=1 upscale models are supported, got {in_n} -> {out_n}.")
    if in_c != 3 or out_c != 3:
        raise ValueError(
            f"Only RGB 3-channel upscale models are supported, got {in_c} -> {out_c}."
        )
    if out_h % in_h != 0 or out_w % in_w != 0:
        raise ValueError(
            "Output shape must be an integer upscale of input, "
            f"got {input_shape} -> {output_shape}."
        )

    scale_h = out_h // in_h
    scale_w = out_w // in_w
    if scale_h != scale_w:
        raise ValueError(f"Non-uniform upscale is not supported, got scale {scale_w}x{scale_h}.")

    return ModelSpec(
        name=name,
        task="upscale",
        inputs=(
            TensorSpec(
                name=input_name,
                layout="nchw",
                dtype="fp32",
                shape=input_shape,
                pixel_format="rgb",
                value_range="0_1",
            ),
        ),
        outputs=(
            TensorSpec(
                name=output_name,
                layout="nchw",
                dtype="fp32",
                shape=output_shape,
                pixel_format="rgb",
                value_range="0_1",
            ),
        ),
        scale=scale_h,
    )
