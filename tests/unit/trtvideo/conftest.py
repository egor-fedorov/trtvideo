import struct
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def make_weighted_onnx():
    def make(path: Path, weight: float = 0.5, *, external: bool = False, dynamic: bool = False):
        shape = [1, 3, "h", "w"] if dynamic else [1, 3, 64, 64]
        scalar = helper.make_tensor("gain", TensorProto.FLOAT, [1], struct.pack("<f", weight), True)
        graph = helper.make_graph(
            [helper.make_node("Mul", ["input", "gain"], ["output"])],
            "gain",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
            [scalar],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save_model(
            model,
            path,
            save_as_external_data=external,
            location="weights.bin",
            size_threshold=0,
        )
        return path

    return make
