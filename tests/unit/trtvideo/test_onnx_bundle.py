import hashlib
from pathlib import Path

import pytest
from onnx import TensorProto, helper

from trtvideo.models.onnx_bundle import OnnxBundleError, inspect_onnx_bundle


def _reference(location: str, *, offset: int = 0) -> TensorProto:
    tensor = TensorProto(name="weight", data_type=TensorProto.UINT8, dims=[1])
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", "1")):
        tensor.external_data.add(key=key, value=value)
    return tensor


def test_identifies_all_external_files_once_without_following_unrelated_files(tmp_path: Path):
    root = tmp_path / "original"
    (root / "data").mkdir(parents=True)
    weights = root / "data/shared.bin"
    weights.write_bytes(b"abcd")
    (root / "sparse.bin").write_bytes(b"s")
    unused = root / "unrelated.bin"
    unused.write_bytes(b"ignored")

    nested = helper.make_graph(
        [helper.make_node("Constant", [], ["c"], value=_reference("data/shared.bin", offset=2))],
        "nested",
        [],
        [],
    )
    graph = helper.make_graph(
        [helper.make_node("Test", [], [], body=nested)],
        "graph",
        [],
        [],
        [_reference("data/shared.bin")],
    )
    graph.sparse_initializer.add().values.CopyFrom(_reference("sparse.bin"))
    path = root / "test.onnx"
    path.write_bytes(helper.make_model(graph).SerializeToString())
    bundle = inspect_onnx_bundle(path)
    assert [item.location for item in bundle.external_files] == ["data/shared.bin", "sparse.bin"]
    assert bundle.external_files[0].sha256 == hashlib.sha256(b"abcd").hexdigest()
    unused.unlink()
    root.rename(tmp_path / "moved")
    path = tmp_path / "moved/test.onnx"
    assert inspect_onnx_bundle(path) == bundle
    weights = path.parent / "data/shared.bin"
    weights.write_bytes(b"abce")  # Whole file, even a byte outside either tensor's slice.
    assert inspect_onnx_bundle(path).identity() != bundle.identity()


@pytest.mark.parametrize(
    "location",
    [
        "../weights",
        "/abs/weights",
        "C:/weights",
        "https://example.test/weights",
        "escape",
        "missing",
    ],
)
def test_rejects_unsafe_or_missing_external_data(tmp_path: Path, location: str):
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "weights"
    outside.write_bytes(b"x")
    (root / "escape").symlink_to(outside)
    graph = helper.make_graph([], "graph", [], [], [_reference(location)])
    path = root / "test.onnx"
    path.write_bytes(helper.make_model(graph).SerializeToString())
    with pytest.raises(OnnxBundleError, match="ONNX external data"):
        inspect_onnx_bundle(path)
