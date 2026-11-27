"""Opt-in compatibility smoke test for external ONNX weights, without downloading models."""

import json
import os
import struct
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnx import TensorProto, helper


def write_model(path: Path, *, dynamic: bool) -> None:
    shape = [1, 3, "H", "W"] if dynamic else [1, 3, 192, 320]
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "gain"], ["y"])],
        "gain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [helper.make_tensor("gain", TensorProto.FLOAT, [1], struct.pack("<f", 0.5), raw=True)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(
        model, path, save_as_external_data=True, location="weights.bin", size_threshold=0
    )


def write_input(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x192:rate=24:duration=5",
            "-vf",
            "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:v",
            "libx264",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
            "-bf",
            "0",
            str(path),
        ],
        check=True,
    )


class ExternalWeightsTest(unittest.TestCase):
    def check_workflow(self, *, dynamic: bool):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "models" / "model.onnx"
            model.parent.mkdir()
            write_model(model, dynamic=dynamic)
            clip = root / "input.mp4"
            write_input(clip)
            output = root / "report"
            command = [
                "trtvideo",
                "compatibility-check",
                "--onnx",
                str(model),
                "--model-name",
                "gain",
                "--model-source",
                "https://example.org/gain",
                "--model-license",
                "CC0-1.0",
                "--output-dir",
                str(output),
                "--input",
                str(clip),
                "--scale",
                "1",
            ]
            # Run outside the model directory: the builder must resolve weights beside the graph.
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-6000:])
            report = json.loads((output / "model-compatibility-report.json").read_text())
            self.assertEqual(report["status"], "valid", report["errors"])
            self.assertEqual(
                report["source_engine_binding"]["build"], "prepared" if dynamic else "direct"
            )
            self.assertEqual(
                report["model"]["source_artifact"]["onnx_bundle"]["external_file_count"], 1
            )
            state = output / "compatibility.state.json"
            before = state.read_bytes()
            resumed = subprocess.run(
                [*command, "--resume"], cwd=root, capture_output=True, text=True
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(state.read_bytes(), before)
            (model.parent / "weights.bin").write_bytes(struct.pack("<f", 0.75))
            changed = subprocess.run(
                [*command, "--resume"], cwd=root, capture_output=True, text=True
            )
            self.assertEqual(changed.returncode, 2, changed.stderr)
            self.assertIn("new --output-dir", changed.stderr)
            self.assertEqual(state.read_bytes(), before)

    def test_static_external_build(self):
        self.check_workflow(dynamic=False)

    @unittest.skipUnless(
        os.environ.get("TRTVIDEO_IMAGE_VARIANT") == "model-tools", "requires model-tools"
    )
    def test_dynamic_external_fp16_preparation(self):
        self.check_workflow(dynamic=True)


if __name__ == "__main__":
    unittest.main()
