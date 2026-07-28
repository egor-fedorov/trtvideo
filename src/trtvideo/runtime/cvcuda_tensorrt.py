"""TensorRT runtime backed by CV-CUDA tensors instead of PyTorch."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import tensorrt as trt

from trtvideo.models.manifest import ModelSpec, TensorDType, make_upscale_model_spec
from trtvideo.runtime import CudaStream, TensorLike
from trtvideo.runtime.cuda import CudaEvent, device_name, set_device

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _cvcuda_dtype_from_trt(dtype: trt.DataType, cvcuda: Any) -> Any:
    if dtype == trt.DataType.FLOAT:
        return cvcuda.Type.F32
    if dtype == trt.DataType.HALF:
        return cvcuda.Type.F16
    raise ValueError(f"Unsupported TensorRT tensor dtype: {dtype}")


def _model_dtype_from_trt(dtype: trt.DataType) -> TensorDType:
    if dtype == trt.DataType.FLOAT:
        return "fp32"
    if dtype == trt.DataType.HALF:
        return "fp16"
    raise ValueError(f"Unsupported TensorRT tensor dtype: {dtype}")


def _tensor_pointer(tensor: Any) -> int:
    interface = tensor.cuda().__cuda_array_interface__
    return int(interface["data"][0])


class CvcudaTensorRTRuntime:
    """Static RGB upscale engine using one CV-CUDA stream and owned GPU tensors."""

    def __init__(
        self,
        engine_path: str,
        quiet: bool = False,
        gpu_id: int = 0,
    ):
        set_device(gpu_id)
        import cvcuda

        self.cvcuda = cvcuda
        self.quiet = quiet
        self.gpu_id = gpu_id
        self.gpu_name = device_name(gpu_id)
        self._trt_runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as engine_file:
            self.engine = self._trt_runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            print(f"ERROR: Failed to load engine: {engine_path}")
            print(
                "  Engine may be built with a different TensorRT version "
                f"(current: {trt.__version__})."
            )
            print("  Rebuild: build-engine <model>.onnx")
            sys.exit(1)

        self.context = self.engine.create_execution_context()
        input_names = []
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "Current video runtime requires exactly one TensorRT input and one output"
            )

        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_trt_dtype = self.engine.get_tensor_dtype(self.input_name)
        self.output_trt_dtype = self.engine.get_tensor_dtype(self.output_name)
        try:
            self.input_dtype = _cvcuda_dtype_from_trt(self.input_trt_dtype, cvcuda)
            self.output_dtype = _cvcuda_dtype_from_trt(self.output_trt_dtype, cvcuda)
            input_model_dtype = _model_dtype_from_trt(self.input_trt_dtype)
            output_model_dtype = _model_dtype_from_trt(self.output_trt_dtype)
        except ValueError as exc:
            print(f"ERROR: Unsupported TensorRT engine contract: {exc}")
            print("  Current video runtime supports fp32/fp16 RGB tensor bindings only.")
            sys.exit(1)

        try:
            self.model_spec: ModelSpec = make_upscale_model_spec(
                name=engine_path,
                input_name=self.input_name,
                output_name=self.output_name,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                input_dtype=input_model_dtype,
                output_dtype=output_model_dtype,
            )
        except ValueError as exc:
            print(f"ERROR: Unsupported TensorRT engine contract: {exc}")
            print("  Current video runtime supports static single-frame RGB upscale engines only.")
            print("  For video inference, build a static engine from a static ONNX variant.")
            sys.exit(1)

        self.input_specs = self.model_spec.inputs
        self.output_specs = self.model_spec.outputs
        _, _, self.input_h, self.input_w = self.input_shape
        _, _, self.output_h, self.output_w = self.output_shape

        self.stream = cvcuda.Stream()
        self.stream_handle = int(self.stream.handle)
        self.gpu_input = cvcuda.Tensor(
            self.input_shape,
            self.input_dtype,
            layout="NCHW",
        )
        self.gpu_output = cvcuda.Tensor(
            self.output_shape,
            self.output_dtype,
            layout="NCHW",
        )
        if not self.context.set_tensor_address(self.input_name, _tensor_pointer(self.gpu_input)):
            raise RuntimeError(f"Failed to bind TensorRT input tensor: {self.input_name}")
        if not self.context.set_tensor_address(self.output_name, _tensor_pointer(self.gpu_output)):
            raise RuntimeError(f"Failed to bind TensorRT output tensor: {self.output_name}")

        if not quiet:
            print(f"  Engine: {engine_path}")
            print(f"  GPU:    {self.gpu_name} (cuda:{gpu_id})")
            print(f"  Input:  {self.input_w}x{self.input_h} {self.model_spec.inputs[0].dtype}")
            print(f"  Output: {self.output_w}x{self.output_h} {self.model_spec.outputs[0].dtype}")
            print(f"  Scale:  {self.model_spec.scale}x")
            print("  Tensor buffers: CV-CUDA (PyTorch-free)")

    def execute(self) -> Any:
        """Enqueue inference against the pre-bound CV-CUDA tensors."""
        if not self.context.execute_async_v3(self.stream_handle):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.gpu_output

    def infer(
        self,
        inputs: Mapping[str, TensorLike],
        *,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> dict[str, TensorLike]:
        """Run raw inference for callers that provide the pre-bound input tensor."""
        if stream is not None and stream is not self.stream:
            raise ValueError("CvcudaTensorRTRuntime uses its owned CV-CUDA stream")
        if inputs.get(self.input_name) is not self.gpu_input:
            raise ValueError("CvcudaTensorRTRuntime expects its preallocated input tensor")
        output = self.execute()
        if synchronize is not False:
            self.synchronize()
        return {self.output_name: output}

    def synchronize(self) -> None:
        """Wait for work queued on the runtime stream."""
        self.stream.sync()

    def create_timing_event(self) -> CudaEvent:
        """Create a CUDA event compatible with ProfileCollector."""
        return CudaEvent()

    def peak_memory_allocated_mb(self) -> float:
        """CV-CUDA does not expose allocator peak usage; benchmark NVML does."""
        return 0.0
