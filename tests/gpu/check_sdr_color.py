"""Compare production model input against independent H.273 patch values."""

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cvcuda
import numpy as np
from check_binding_strides import _build_identity, _copy_planes, cudart, trt

from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
from trtvideo.video.color import SdrColorContract
from trtvideo.video.nvcodec.frame_processor import NvcodecFrameProcessor


class SdrColorTest(unittest.TestCase):
    def test_model_input(self):
        patches = [(16, 128, 128), (235, 128, 128), (128, 180, 100), (92, 85, 215)]
        width, height = 64, 32
        nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
        for idx, (y, u, v) in enumerate(patches):
            patch = nv12[:, idx * 16 : idx * 16 + 16]
            patch[:height] = y
            patch[height:, ::2] = u
            patch[height:, 1::2] = v
        with TemporaryDirectory() as directory:
            engine = Path(directory, "identity.engine")
            shape = (1, 3, height, width)
            _build_identity(engine, shape, trt.float32)
            runtime = CvcudaTensorRTRuntime(str(engine), quiet=True)
            src = cvcuda.Tensor((1, *nv12.shape, 1), cvcuda.Type.U8, layout="NHWC", rowalign=1)
            ptr = src.cuda().__cuda_array_interface__["data"][0]
            (status,) = cudart.cudaMemcpy(
                ptr, nv12.ctypes.data, nv12.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            self.assertEqual(status, cudart.cudaError_t.cudaSuccess)
            for matrix, kr, kb in (
                ("bt470bg", 0.299, 0.114),
                ("bt709", 0.2126, 0.0722),
                ("bt2020nc", 0.2627, 0.0593),
            ):
                for limited in (False, True):
                    with self.subTest(matrix=matrix, limited=limited):
                        contract = SdrColorContract(
                            "tv" if limited else "pc", matrix, "bt709", "bt709"
                        )
                        processor = NvcodecFrameProcessor(
                            runtime,
                            color_spec_name=contract.cvcuda_spec_name,
                            limited_range=limited,
                        )
                        processor.preprocess(src)
                        runtime.synchronize()
                        actual = np.empty(shape, dtype=np.float32)
                        _copy_planes(runtime.gpu_input, actual, upload=False)
                        for idx, (y, u, v) in enumerate(patches):
                            lum = (y - 16) * 255 / 219 if limited else y
                            cb = (u - 128) * (255 / 224 if limited else 1)
                            cr = (v - 128) * (255 / 224 if limited else 1)
                            r = lum + 2 * (1 - kr) * cr
                            b = lum + 2 * (1 - kb) * cb
                            g = (lum - kr * r - kb * b) / (1 - kr - kb)
                            expected = np.rint(np.clip([r, g, b], 0, 255))
                            np.testing.assert_allclose(
                                actual[0, :, 8, 16 * idx + 8] * 255, expected, rtol=0, atol=2
                            )

    def test_publication_and_early_rejection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "identity.engine"
            _build_identity(engine, (1, 3, 192, 320), trt.float32)
            for matrix in ("bt2020nc", "smpte240m"):
                source = root / f"{matrix}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=size=320x192:rate=24",
                        "-frames:v",
                        "8",
                        "-c:v",
                        "libx264",
                        "-colorspace",
                        matrix,
                        "-color_range",
                        "tv",
                        str(source),
                    ],
                    check=True,
                )
                for codec in ("h264", "hevc"):
                    output = root / "out.mp4"
                    output.write_bytes(b"previous output")
                    proc = subprocess.run(
                        [
                            "trtvideo",
                            "--engine",
                            str(engine),
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--codec",
                            codec,
                            "--bitrate-mbps",
                            "2",
                            "--max-frames",
                            "4",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if matrix == "smpte240m":
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn(matrix, proc.stderr)
                        self.assertNotIn("Initializing TensorRT", proc.stderr)
                        self.assertEqual(output.read_bytes(), b"previous output")
                        continue
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    raw = root / f"out.{codec}"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            str(output),
                            "-c:v",
                            "copy",
                            "-f",
                            codec,
                            str(raw),
                        ],
                        check=True,
                    )
                    for path in (output, raw):
                        probe = subprocess.check_output(
                            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
                        )
                        stream = json.loads(probe)["streams"][0]
                        self.assertEqual(
                            (stream["color_space"], stream["color_range"]), (matrix, "tv")
                        )


if __name__ == "__main__":
    unittest.main()
