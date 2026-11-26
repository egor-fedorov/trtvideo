from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

from trtvideo.cli import build_engine, prepare_onnx
from trtvideo.cli.export_onnx import (
    PIXEL_UNSHUFFLE_TRANSPOSE_PERMUTATION,
    _validate_channel_major_unshuffle_graph,
    build_parser,
    export_filename,
    export_filename_for_target,
)
from trtvideo.cli.prepare_onnx import (
    ONNXPrecision,
    convert_to_mixed_precision,
    is_dynamic,
    output_path_for_variant,
    parse_size,
)
from trtvideo.models.onnx_bundle import OnnxBundleError, inspect_onnx_bundle
from trtvideo.models.onnx_preparation import preparation_path, read_preparation


def test_preparation_records_each_target_from_original_bundle(
    tmp_path, make_weighted_onnx, monkeypatch
):
    source = make_weighted_onnx(tmp_path / "source.onnx", external=True, dynamic=True)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare-onnx",
            str(source),
            "--scale",
            "1",
            "--size",
            "32x32",
            "--size",
            "64x48",
            "--output_dir",
            str(output),
        ],
    )
    prepare_onnx.main()
    before = inspect_onnx_bundle(source).identity()
    for width, height in ((32, 32), (64, 48)):
        graph = output / f"source_{width}x{height}.onnx"
        record = read_preparation(graph, inspect_onnx_bundle(graph).identity())
        assert record["source"] == before
        assert (record["width"], record["height"], record["scale"]) == (width, height, 1)
        assert record["precision"] == "fp32"
        assert record["source"]["external_file_count"] == 1
        assert record["output"]["external_file_count"] == 0
    make_weighted_onnx(graph, weight=0.75)
    monkeypatch.setattr(
        build_engine, "_load_tensorrt", lambda: pytest.fail("must not start builder")
    )
    with pytest.raises(OnnxBundleError, match="rerun prepare-onnx"):
        build_engine.build_engine(str(graph), str(tmp_path / "model.engine"))


def test_failed_conversion_does_not_leave_old_preparation_receipt(
    tmp_path, make_weighted_onnx, monkeypatch
):
    source = make_weighted_onnx(tmp_path / "source.onnx")
    output = tmp_path / "source_fp16.onnx"
    preparation_path(output).write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare-onnx", str(source), "--output_dir", str(tmp_path), "--precision", "fp16"],
    )

    def fail(src, out):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(prepare_onnx, "convert_to_mixed_precision", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        prepare_onnx.main()
    assert not preparation_path(output).exists()


def test_export_filename_uses_explicit_model_name() -> None:
    assert export_filename("liveaction_span", 1080) == "liveaction_span_1080p.onnx"


def test_export_filename_for_custom_target() -> None:
    assert (
        export_filename_for_target(
            "model",
            {"name": "640x360", "h": 360, "w": 640},
        )
        == "model_640x360.onnx"
    )


def test_export_parser_defaults_to_all_standard_sizes() -> None:
    args = build_parser().parse_args([])

    assert args.size == []


def test_export_parser_accepts_selected_sizes() -> None:
    args = build_parser().parse_args(["--size", "1280x720", "--size", "640x360"])

    assert args.size == ["1280x720", "640x360"]


def test_export_parser_accepts_explicit_conformance_report() -> None:
    args = build_parser().parse_args(["--conformance-report", "/tmp/model-report.json"])

    assert args.conformance_report == Path("/tmp/model-report.json")


def test_pixel_unshuffle_transposes_spatial_offsets_after_channels() -> None:
    assert PIXEL_UNSHUFFLE_TRANSPOSE_PERMUTATION == (0, 1, 3, 5, 2, 4)


def test_channel_major_export_rejects_onnx_space_to_depth() -> None:
    node = type("Node", (), {"op_type": "SpaceToDepth"})()
    graph = type("Graph", (), {"node": [node]})()
    model = type("Model", (), {"graph": graph})()

    with pytest.raises(RuntimeError, match="incompatible channel ordering"):
        _validate_channel_major_unshuffle_graph(model)


def test_fp16_conversion_suppresses_only_expected_truncation_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = object()
    converted_model = object()
    saved: list[tuple[object, str]] = []

    onnx_module = ModuleType("onnx")
    onnx_module.__dict__["load"] = lambda _path: source_model
    onnx_module.__dict__["save"] = lambda model, path: saved.append((model, path))

    def convert(model: object, *, keep_io_types: bool) -> object:
        assert model is source_model
        assert keep_io_types is True
        warnings.warn(
            "the float32 number 4e-08 will be truncated to 1e-07",
            UserWarning,
            stacklevel=1,
        )
        warnings.warn("actionable converter warning", UserWarning, stacklevel=1)
        return converted_model

    float16_module = ModuleType("onnxconverter_common.float16")
    float16_module.__dict__["convert_float_to_float16"] = convert
    converter_package = ModuleType("onnxconverter_common")
    converter_package.__dict__["float16"] = float16_module
    monkeypatch.setitem(sys.modules, "onnx", onnx_module)
    monkeypatch.setitem(sys.modules, "onnxconverter_common", converter_package)

    output = tmp_path / "model_fp16.onnx"
    with pytest.warns(UserWarning, match="actionable converter warning") as emitted:
        convert_to_mixed_precision("model.onnx", str(output))

    assert len(emitted) == 1
    assert saved == [(converted_model, str(output))]


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
