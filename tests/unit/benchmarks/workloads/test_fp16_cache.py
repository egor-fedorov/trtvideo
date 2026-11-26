import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from benchmarks.scripts.workloads import prepare
from trtvideo.models import onnx_preparation
from trtvideo.models.onnx_bundle import inspect_onnx_bundle


def graph(path, width, *, fp16, weight=1.0):
    dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    tensor = helper.make_tensor("weight", dtype, [1], [weight])
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, width])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, width])
    nodes = [
        helper.make_node("Cast", ["weight"], ["fp32"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["x", "fp32"], ["y"]),
    ]
    model = helper.make_model(helper.make_graph(nodes, "tiny", [value], [output], [tensor]))
    helper.set_model_props(
        model,
        {
            prepare.EXPORT_SCALE_METADATA_KEY: "1",
            prepare.EXPORT_CONTRACT_METADATA_KEY: prepare.EXPORT_CONTRACT_METADATA_VALUE,
        },
    )
    onnx.save(model, str(path))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    converter = dict(policy_version=1, onnx="1.23", onnxconverter_common="1.16", keep_io_types=True)
    monkeypatch.setattr(onnx_preparation, "current_fp16_converter", lambda: converter)
    monkeypatch.setattr(prepare, "current_fp16_converter", lambda: converter)
    variants = [
        dict(
            name=str(w),
            input_width=w,
            input_height=8,
            fp32_path=f"m{w}.onnx",
            fp16_path=f"m{w}_fp16.onnx",
        )
        for w in (8, 16)
    ]
    manifest = {
        "model": {
            "variants": variants,
            "onnx_dir": ".",
            "export_name": "tiny",
            "scale": 1,
            "export_conformance": {"report_path": "exports.json"},
            "weights_path": "tiny.pth",
        }
    }
    exports = []

    def convert(variant):
        src, dst = (tmp_path / variant[k] for k in ("fp32_path", "fp16_path"))
        with onnx_preparation.record_preparation(
            src,
            inspect_onnx_bundle(src).identity(),
            dst,
            width=variant["input_width"],
            height=8,
            scale=1,
            precision="fp16",
        ):
            graph(dst, variant["input_width"], fp16=True)

    for v in variants:
        src = tmp_path / v["fp32_path"]
        graph(src, v["input_width"], fp16=False)
        exports.append({"path": src.name, "onnx_bundle": inspect_onnx_bundle(src).identity()})
        convert(v)
    evidence = {"exports": exports}
    monkeypatch.setattr(prepare, "verify_export_conformance", lambda *_: evidence)
    calls = []

    def run(command):
        assert command[0] == "prepare-onnx"
        calls.append(Path(command[1]).name)
        convert(next(v for v in variants if v["fp32_path"] == calls[-1]))

    monkeypatch.setattr(prepare, "run_command", run)
    return manifest, evidence, calls


@pytest.mark.parametrize(
    "fault", ["weight", "source", "missing", "legacy", "version", "policy", "io", "width"]
)
def test_verify_rejects_unproven_cache_and_rebuilds_only_one_variant(cache, tmp_path, fault):
    manifest, evidence, calls = cache
    model = manifest["model"]
    v = model["variants"][0]
    dst = tmp_path / v["fp16_path"]
    receipt = onnx_preparation.preparation_path(dst)
    prepare.prepare_model(manifest, tmp_path, force=False)
    assert not calls
    doc = json.loads(receipt.read_text())
    if fault == "weight":
        graph(dst, 8, fp16=True, weight=0.5)
    elif fault == "missing":
        receipt.unlink()
    else:
        if fault == "legacy":
            doc.pop("converter")
        elif fault == "version":
            doc["converter"]["onnxconverter_common"] = "1.15"
        elif fault == "policy":
            doc["converter"]["policy_version"] = 9
        elif fault == "io":
            doc["converter"]["keep_io_types"] = False
        elif fault == "width":
            doc["width"] = 16
        else:
            doc["source"] = evidence["exports"][1]["onnx_bundle"]
        receipt.write_text(json.dumps(doc))
    with pytest.raises(prepare.WorkloadError, match="rerun prepare"):
        prepare.verified_variant(model, v, tmp_path, evidence)
    assert not calls
    prepare.prepare_model(manifest, tmp_path, force=False)
    prepare.prepare_model(manifest, tmp_path, force=False)
    assert calls == [v["fp32_path"]]
    entry = prepare.verified_variant(model, v, tmp_path, evidence)
    assert entry["preparation_sha256"] == prepare.sha256_file(receipt)
    assert entry["sha256"] == entry["preparation"]["output"]["graph_sha256"]


def test_verify_assets_never_locks_or_repairs_an_unproven_variant(cache, tmp_path):
    manifest, _, calls = cache
    source = tmp_path / "tiny.pth"
    source.write_bytes(b"test input")
    identity = {
        "sha256": prepare.sha256_file(source),
        "size_bytes": source.stat().st_size,
        "attribution": "test",
    }
    manifest["model"]["source"] = identity
    manifest.update(
        id="test", clip={"source_path": source.name, "source": identity, "variants": []}
    )
    valid = prepare.verify_assets(manifest, tmp_path)
    assert len(valid["assets"]) == 2 and all("preparation_sha256" in a for a in valid["assets"])
    receipt = onnx_preparation.preparation_path(tmp_path / "m8_fp16.onnx")
    receipt.unlink()
    with pytest.raises(prepare.WorkloadError, match="rerun prepare"):
        prepare.verify_assets(manifest, tmp_path)
    assert not receipt.exists() and not calls


def test_fp32_conformance_binds_external_weights(tmp_path, monkeypatch):
    import numpy as np

    graph_path = tmp_path / "m.onnx"
    graph(graph_path, 8, fp16=False)
    model = onnx.load(graph_path)
    model.graph.initializer[0].CopyFrom(
        onnx.numpy_helper.from_array(np.array([1.0], dtype="<f4"), "weight")
    )
    onnx.save_model(
        model, graph_path, save_as_external_data=True, location="weights.bin", size_threshold=0
    )
    identity = inspect_onnx_bundle(graph_path).identity()
    entry = {
        "path": graph_path.name,
        "onnx_bundle": identity,
        "sha256": identity["graph_sha256"],
        "size_bytes": identity["graph_size_bytes"],
    }
    report = tmp_path / "exports.json"
    report.write_text(json.dumps({"exports": [entry], "tools": {}}))
    model_config = {
        "export_name": "m",
        "onnx_dir": ".",
        "scale": 1,
        "source": {"sha256": "a" * 64, "size_bytes": 4},
        "export_conformance": {"report_path": report.name},
        "variants": [{"fp32_path": graph_path.name}],
    }
    monkeypatch.setattr(prepare, "validate_conformance_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prepare, "export_tool_versions", lambda: {})
    assert prepare.verify_export_conformance({"model": model_config}, tmp_path)["exports"] == [
        entry
    ]
    (tmp_path / "weights.bin").write_bytes(np.array([2.0], dtype="<f4").tobytes())
    with pytest.raises(prepare.WorkloadError, match="bundle changed"):
        prepare.verify_export_conformance({"model": model_config}, tmp_path)
    del entry["onnx_bundle"]
    report.write_text(json.dumps({"exports": [entry], "tools": {}}))
    with pytest.raises(prepare.WorkloadError, match="external weights"):
        prepare.verify_export_conformance({"model": model_config}, tmp_path)
