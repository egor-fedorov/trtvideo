import pytest

from trtvideo.models.manifest import make_upscale_model_spec
from trtvideo.runtime.bindings import TensorBindingError, validate_binding_strides

MODEL = make_upscale_model_spec(
    name="mixed-io",
    input_name="source",
    output_name="result",
    input_shape=(1, 3, 2, 5),
    output_shape=(1, 3, 4, 10),
    input_dtype="fp32",
    output_dtype="fp16",
)


@pytest.mark.parametrize(
    ("spec", "trt_strides", "byte_strides"),
    [
        (MODEL.inputs[0], (30, 10, 5, 1), (120, 40, 20, 4)),
        (MODEL.outputs[0], (120, 40, 10, 1), (240, 80, 20, 2)),
    ],
)
def test_linear_binding_accepts_packed_or_implicit_strides(spec, trt_strides, byte_strides):
    for actual in (byte_strides, None, (0, *byte_strides[1:])):
        validate_binding_strides(spec, trt_element_strides=trt_strides, cuda_byte_strides=actual)


@pytest.mark.parametrize(
    ("spec", "trt_strides", "actual", "expected"),
    [
        (MODEL.inputs[0], (30, 10, 5, 1), (144, 48, 24, 4), (120, 40, 20, 4)),
        (MODEL.outputs[0], (120, 40, 10, 1), (288, 96, 24, 2), (240, 80, 20, 2)),
    ],
)
def test_binding_rejects_row_padding_with_actionable_error(spec, trt_strides, actual, expected):
    with pytest.raises(TensorBindingError) as exc:
        validate_binding_strides(spec, trt_element_strides=trt_strides, cuda_byte_strides=actual)
    message = str(exc.value)
    for detail in (spec.name, spec.dtype, str(spec.shape), str(actual), str(expected)):
        assert detail in message


@pytest.mark.parametrize(
    ("trt_strides", "cuda_strides"),
    [
        ((36, 12, 6, 1), None),  # A packed buffer cannot satisfy pitched engine strides.
        ((30, 10, 5, 1), (120, 44, 20, 4)),  # Matching row stride but wrong channel stride.
        ((30, 10, 5, 1), (40, 20, 4)),  # A missing axis is not a singleton stride.
    ],
)
def test_binding_compares_all_axes(trt_strides, cuda_strides):
    with pytest.raises(TensorBindingError):
        validate_binding_strides(
            MODEL.inputs[0], trt_element_strides=trt_strides, cuda_byte_strides=cuda_strides
        )
