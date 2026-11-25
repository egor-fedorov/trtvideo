"""CPU integration coverage requiring the complete model-tools dependencies."""

from pathlib import Path

import numpy as np
import pytest

from trtvideo.cli.export_onnx import (
    _pixel_unshuffle_translation_table,
    _translate_pixel_unshuffle,
    _validate_channel_major_unshuffle_graph,
)


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("channels", [1, 3])
def test_pixel_unshuffle_eager_matches_torch(factor: int, channels: int) -> None:
    import torch

    inputs = np.arange(channels * 8 * 12, dtype=np.float32).reshape(1, channels, 8, 12)
    expected = torch.nn.functional.pixel_unshuffle(torch.from_numpy(inputs), factor).numpy()

    actual = _translate_pixel_unshuffle(inputs, factor)

    np.testing.assert_array_equal(actual.value, expected)


@pytest.mark.parametrize("factor", [2, 4])
def test_pixel_unshuffle_export_matches_torch(tmp_path: Path, factor: int) -> None:
    import onnx
    import onnxruntime
    import torch

    model = torch.nn.PixelUnshuffle(factor).eval()
    inputs = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(1, 3, 8, 12)
    program = torch.onnx.export(
        model,
        (inputs,),
        f=None,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        custom_translation_table=_pixel_unshuffle_translation_table(torch),
    )
    assert program is not None
    graph = program.model_proto
    onnx.checker.check_model(graph)
    _validate_channel_major_unshuffle_graph(graph)
    path = tmp_path / "pixel-unshuffle.onnx"
    onnx.save(graph, path, convert_attribute=True)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    (actual,) = session.run(None, {"input": inputs.numpy()})

    np.testing.assert_array_equal(actual, model(inputs).numpy())
