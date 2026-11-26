"""Opt-in identity-engine regression for the production CV-CUDA/TensorRT bindings."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cvcuda
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from trtvideo.runtime.bindings import TensorBindingError
from trtvideo.runtime.cvcuda_tensorrt import TRT_LOGGER, CvcudaTensorRTRuntime


def _build_identity(
    path: Path,
    shape: tuple[int, ...],
    dtype: trt.DataType,
    *,
    output_dtype: trt.DataType | None = None,
    output_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    source = network.add_input("source", dtype, shape)
    layer = (
        network.add_identity(source)
        if output_dtype is None or output_dtype == dtype
        else network.add_cast(source, output_dtype)
    )
    output = layer.get_output(0)
    output.name = "result"
    network.mark_output(output)
    output.allowed_formats = 1 << int(output_format)
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    assert engine is not None, "Failed to build the identity engine"
    path.write_bytes(bytes(engine))


def _copy_planes(tensor: cvcuda.Tensor, host: np.ndarray, *, upload: bool) -> None:
    interface = tensor.cuda().__cuda_array_interface__
    assert host.flags.c_contiguous and tuple(interface["shape"]) == host.shape
    assert np.dtype(interface["typestr"]) == host.dtype
    _, plane_pitch, row_pitch, pixel_stride = interface.get("strides") or host.strides
    assert pixel_stride == host.itemsize
    for channel, plane in enumerate(host[0]):
        device_plane = int(interface["data"][0]) + channel * plane_pitch
        device = (device_plane, row_pitch)
        cpu = (plane.ctypes.data, plane.strides[0])
        if upload:
            dst, src, kind = device, cpu, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        else:
            dst, src, kind = cpu, device, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        (status,) = cudart.cudaMemcpy2D(*dst, *src, plane.strides[0], plane.shape[0], kind)
        assert status == cudart.cudaError_t.cudaSuccess, f"cudaMemcpy2D failed: {status}"


class BindingStridesTest(unittest.TestCase):
    def _check_widths(self, input_dtype: trt.DataType, output_dtype: trt.DataType) -> None:
        for width in (854, 1280):
            with self.subTest(width=width), TemporaryDirectory() as directory:
                shape = (1, 3, 480, width)
                path = Path(directory, "identity.engine")
                _build_identity(path, shape, input_dtype, output_dtype=output_dtype)
                runtime = CvcudaTensorRTRuntime(str(path), quiet=True)
                values = np.random.default_rng(0).integers(1024, size=shape, dtype=np.int16)
                dtype = trt.nptype(input_dtype)
                source = values.astype(dtype) / dtype(1024)
                expected = source.astype(trt.nptype(output_dtype), copy=False)
                # Mark unwritten pixels even if both bindings have the same padding.
                actual = np.full_like(expected, -1)
                _copy_planes(runtime.gpu_input, source, upload=True)
                _copy_planes(runtime.gpu_output, actual, upload=True)
                (status,) = cudart.cudaDeviceSynchronize()
                self.assertEqual(status, cudart.cudaError_t.cudaSuccess)

                result = runtime.execute()
                runtime.synchronize()
                _copy_planes(result, actual, upload=False)
                np.testing.assert_array_equal(actual, expected)

    def test_fp32_identity(self) -> None:
        self._check_widths(trt.float32, trt.float32)

    def test_rejects_nonlinear_output(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "hwc.engine")
            _build_identity(path, (1, 3, 64, 64), trt.float32, output_format=trt.TensorFormat.HWC)
            with self.assertRaisesRegex(TensorBindingError, "binding 'result'.*LINEAR.*HWC"):
                CvcudaTensorRTRuntime(str(path), quiet=True)

    def test_fp16_identity(self) -> None:
        self._check_widths(trt.float16, trt.float16)

    def test_mixed_bindings(self) -> None:
        for src, dst in ((trt.float16, trt.float32), (trt.float32, trt.float16)):
            with self.subTest(src=src, dst=dst):
                self._check_widths(src, dst)


if __name__ == "__main__":
    unittest.main()
