import os

import pytest

from ai_media.cli.export_onnx import export_filename
from ai_media.cli.prepare_onnx import ONNXPrecision, is_dynamic, output_path_for_variant, parse_size


def test_export_filename_uses_explicit_model_name() -> None:
    assert export_filename("liveaction_span", 1080) == "liveaction_span_1080p.onnx"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1280x720", {"name": "720p", "h": 720, "w": 1280}),
        ("1920*1080", {"name": "1080p", "h": 1080, "w": 1920}),
        ("640x360", {"name": "640x360", "h": 360, "w": 640}),
    ],
)
def test_parse_size(value: str, expected: dict[str, int | str]) -> None:
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["bad", "0x720", "1280x0", "-1x720"])
def test_parse_size_exits_for_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_size(value)


@pytest.mark.parametrize(
    ("dims", "expected"),
    [
        ([1, 3, 720, 1280], False),
        ([1, 3, "height", 1280], True),
        ([1, 3, -1, 1280], True),
        ([1, 3, 0, 1280], True),
    ],
)
def test_is_dynamic(dims: list[int | str], expected: bool) -> None:
    assert is_dynamic(dims) is expected


@pytest.mark.parametrize(
    ("variant", "precision", "expected"),
    [
        ("720p", "fp32", "model_720p.onnx"),
        ("720p", "fp16", "model_720p_fp16.onnx"),
        (None, "fp16", "model_fp16.onnx"),
    ],
)
def test_output_path_for_variant(
    variant: str | None,
    precision: ONNXPrecision,
    expected: str,
) -> None:
    assert output_path_for_variant("/tmp/onnx", "model", variant, precision) == os.path.join(
        "/tmp/onnx",
        expected,
    )
