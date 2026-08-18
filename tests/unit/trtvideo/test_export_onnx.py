from types import SimpleNamespace

import pytest

from trtvideo.cli.export_onnx import _validate_exported_graph_contract


def _tensor(shape: list[int], *, dynamic_index: int | None = None) -> SimpleNamespace:
    dimensions = []
    for index, value in enumerate(shape):
        dimensions.append(
            SimpleNamespace(
                dim_param="dynamic" if index == dynamic_index else "",
                dim_value=0 if index == dynamic_index else value,
            )
        )
    return SimpleNamespace(
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(
                shape=SimpleNamespace(dim=dimensions),
            )
        )
    )


def _model(input_shape: list[int], output_shape: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        graph=SimpleNamespace(
            input=[_tensor(input_shape)],
            output=[_tensor(output_shape)],
        )
    )


def test_exported_graph_accepts_probe_inferred_x4_scale() -> None:
    _validate_exported_graph_contract(
        _model([1, 3, 720, 1280], [1, 3, 2880, 5120]),
        input_h=720,
        input_w=1280,
        scale=4,
    )


def test_exported_graph_rejects_scale_that_changes_with_resolution() -> None:
    with pytest.raises(RuntimeError, match="changed from probe 4x to 2x"):
        _validate_exported_graph_contract(
            _model([1, 3, 720, 1280], [1, 3, 1440, 2560]),
            input_h=720,
            input_w=1280,
            scale=4,
        )


def test_exported_graph_rejects_dynamic_shape() -> None:
    model = _model([1, 3, 720, 1280], [1, 3, 2880, 5120])
    model.graph.output[0] = _tensor([1, 3, 2880, 5120], dynamic_index=2)

    with pytest.raises(RuntimeError, match="dynamic tensor shape"):
        _validate_exported_graph_contract(
            model,
            input_h=720,
            input_w=1280,
            scale=4,
        )
