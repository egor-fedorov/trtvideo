"""ONNX metadata checks against real protobuf types; no TensorRT or GPU needed."""

import pytest

from trtvideo.cli.build_engine import _onnx_tensor_elem_type, infer_onnx_precision


@pytest.mark.parametrize(
    ("binding_type", "weight_type", "expected"),
    [
        ("FLOAT", "FLOAT", "fp32"),
        ("FLOAT16", "FLOAT16", "fp16"),
        ("FLOAT", "FLOAT16", "fp16"),
        ("INT64", "INT64", "unknown"),
    ],
)
def test_onnx_precision_uses_bindings_and_initializers(
    tmp_path,
    binding_type,
    weight_type,
    expected,
) -> None:
    import onnx

    binding_dtype = getattr(onnx.TensorProto, binding_type)
    weight_dtype = getattr(onnx.TensorProto, weight_type)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "precision probe",
        [onnx.helper.make_tensor_value_info("input", binding_dtype, [1, 3, 2, 2])],
        [onnx.helper.make_tensor_value_info("output", binding_dtype, [1, 3, 2, 2])],
        [onnx.helper.make_tensor("weights", weight_dtype, [1], [0])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), path)

    assert infer_onnx_precision(str(path)) == expected


def test_non_tensor_and_unset_onnx_values_have_no_tensor_element_type() -> None:
    import onnx

    unset = onnx.ValueInfoProto(name="unset")
    sequence = onnx.helper.make_tensor_sequence_value_info("sequence", onnx.TensorProto.FLOAT, [1])
    tensor = onnx.helper.make_tensor_value_info("tensor", onnx.TensorProto.FLOAT, [1])

    assert _onnx_tensor_elem_type(unset) is None
    assert _onnx_tensor_elem_type(sequence) is None
    assert _onnx_tensor_elem_type(tensor) == onnx.TensorProto.FLOAT
