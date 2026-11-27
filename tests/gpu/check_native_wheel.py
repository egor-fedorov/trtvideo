"""Exercise the installed native GPU wheel, including its ONNX builder and muxer."""

import json
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from check_onnx_provenance import write_input, write_model


class NativeWheelTest(unittest.TestCase):
    def test_installed_gpu_profile(self):
        self.assertIsNone(find_spec("torch"), "the gpu extra should not install PyTorch")
        scripts = Path(sys.executable).parent
        with TemporaryDirectory() as directory:
            work = Path(directory)
            write_model(work / "model.onnx", dynamic=False)
            write_input(work / "input.mp4")
            for command in (
                [scripts / "trtvideo", "doctor"],
                [scripts / "build-engine", "model.onnx", "--output", "model.engine"],
                [
                    scripts / "trtvideo",
                    "--engine",
                    "model.engine",
                    "--input",
                    "input.mp4",
                    "--output",
                    "output.mp4",
                    "--max-frames",
                    "8",
                ],
            ):
                subprocess.run(command, check=True, cwd=work)
            data = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_streams",
                        "-show_packets",
                        "-of",
                        "json",
                        "output.mp4",
                    ],
                    cwd=work,
                )
            )
            (stream,) = data["streams"]
            self.assertEqual(
                (stream["width"], stream["height"], stream["codec_name"]), (320, 192, "h264")
            )
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(stream["color_range"], "tv")
            for key in ("color_space", "color_transfer", "color_primaries"):
                self.assertEqual(stream[key], "bt709")
            self.assertGreater(int(stream["bit_rate"]), 0)
            self.assertAlmostEqual(float(stream["duration"]), 8 / 24, places=5)
            pts = [int(packet["pts"]) for packet in data["packets"]]
            self.assertEqual(len(pts), 8)
            self.assertEqual(pts, sorted(set(pts)))


if __name__ == "__main__":
    unittest.main()
