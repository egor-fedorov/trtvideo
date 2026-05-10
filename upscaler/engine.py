"""TensorRT inference wrapper with GPU-side pre/postprocessing."""

import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import tensorrt as trt
import torch

from upscaler.model_spec import ModelSpec, make_upscale_model_spec
from upscaler.runtime import CudaStream, TensorLike

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTRuntime:
    """TensorRT inference with GPU-side pre/postprocessing.

    Pipeline: CPU uint8 -> GPU -> float32/permute -> TRT -> permute/uint8 -> CPU.
    """

    def __init__(self, engine_path: str, quiet: bool = False, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            print(f"ERROR: Failed to load engine: {engine_path}")
            print(
                "  Engine may be built with a different TensorRT version "
                f"(current: {trt.__version__})."
            )
            print("  Rebuild: build-engine <model>.onnx")
            sys.exit(1)

        self.context = self.engine.create_execution_context()

        # Determine shapes
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        try:
            self.model_spec: ModelSpec = make_upscale_model_spec(
                name=engine_path,
                input_name=self.input_name,
                output_name=self.output_name,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
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

        # Preallocate GPU tensors (reused every frame)
        self.gpu_input = torch.empty(self.input_shape, dtype=torch.float32, device=self.device)
        self.gpu_output = torch.empty(self.output_shape, dtype=torch.float32, device=self.device)

        # Bind GPU buffers to TensorRT context
        self.context.set_tensor_address(self.input_name, self.gpu_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.gpu_output.data_ptr())

        # CUDA stream (shared between torch and TRT)
        self.stream = torch.cuda.Stream(device=self.device)

        if not quiet:
            print(f"  Engine: {engine_path}")
            print(f"  GPU:    cuda:{gpu_id}")
            print(f"  Input:  {self.input_w}x{self.input_h}")
            print(f"  Output: {self.output_w}x{self.output_h}")
            print(f"  Scale:  {self.model_spec.scale}x")

    def infer(
        self,
        inputs: Mapping[str, TensorLike],
        *,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> dict[str, TensorLike]:
        """Run raw TensorRT inference on prepared NCHW GPU tensors.

        If a stream is provided, the caller owns synchronization. Otherwise the runtime uses
        its internal stream and synchronizes before returning.
        """
        if self.input_name not in inputs:
            raise KeyError(f"Missing runtime input tensor: {self.input_name}")

        run_stream = stream if stream is not None else self.stream
        should_sync = stream is None if synchronize is None else synchronize

        output = self._execute_nchw(inputs[self.input_name], run_stream)

        if should_sync:
            run_stream.synchronize()

        return {self.output_name: output}

    def infer_rgb_cpu(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Upscale a single frame (CPU->GPU->TRT->GPU->CPU).

        Args:
            frame_rgb: Input frame (H, W, 3) uint8 RGB.

        Returns:
            Upscaled frame (H*scale, W*scale, 3) uint8 RGB.
        """
        with torch.cuda.stream(self.stream):
            frame_gpu = torch.from_numpy(frame_rgb).to(device=self.device, non_blocking=True)
            frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            output_nchw = self._execute_nchw(frame_gpu, self.stream)
            output = self._output_nchw_to_rgb(output_nchw)

        self.stream.synchronize()
        return output.cpu().numpy()

    def infer_rgb_cpu_profiled(
        self,
        frame_rgb: np.ndarray,
        events: tuple[Any, Any, Any, Any],
    ) -> np.ndarray:
        """Upscale a CPU RGB frame and record preprocess/TRT/postprocess CUDA events."""
        e0, e1, e2, e3 = events
        with torch.cuda.stream(self.stream):
            e0.record(self.stream)
            frame_gpu = torch.from_numpy(frame_rgb).to(device=self.device, non_blocking=True)
            frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            e1.record(self.stream)
            output_nchw = self._execute_nchw(frame_gpu, self.stream)
            e2.record(self.stream)
            output = self._output_nchw_to_rgb(output_nchw)
            e3.record(self.stream)

        self.stream.synchronize()
        return output.cpu().numpy()

    def infer_rgb_tensor(
        self,
        rgb_hwc: TensorLike,
        *,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> TensorLike:
        """Upscale a GPU RGB HWC uint8 tensor and return a GPU RGB HWC uint8 tensor."""
        run_stream = stream if stream is not None else self.stream
        should_sync = stream is None if synchronize is None else synchronize

        with torch.cuda.stream(run_stream):
            input_nchw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            output_nchw = self._execute_nchw(input_nchw, run_stream)
            output = self._output_nchw_to_rgb(output_nchw)

        if should_sync:
            run_stream.synchronize()

        return output

    def infer_rgb_tensor_into(
        self,
        rgb_hwc: TensorLike,
        output_rgb_hwc: TensorLike,
        *,
        input_nchw: TensorLike | None = None,
        output_rgb_float: TensorLike | None = None,
        stream: CudaStream | None = None,
        synchronize: bool | None = None,
    ) -> TensorLike:
        """Upscale a GPU RGB HWC uint8 tensor into a preallocated RGB HWC uint8 tensor."""
        run_stream = stream if stream is not None else self.stream
        should_sync = stream is None if synchronize is None else synchronize

        with torch.cuda.stream(run_stream):
            prepared_input = self.gpu_input if input_nchw is None else input_nchw
            prepared_input.copy_(rgb_hwc.permute(2, 0, 1).unsqueeze(0))
            prepared_input.div_(255.0)
            output_nchw = self._execute_nchw(prepared_input, run_stream)
            output = self._output_nchw_to_rgb_into(
                output_nchw,
                output_rgb_hwc,
                scratch_hwc=output_rgb_float,
            )

        if should_sync:
            run_stream.synchronize()

        return output

    def _execute_nchw(self, input_nchw: TensorLike, stream: CudaStream) -> TensorLike:
        with torch.cuda.stream(stream):
            if input_nchw.data_ptr() != self.gpu_input.data_ptr():
                self.gpu_input.copy_(input_nchw)
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        return self.gpu_output

    @staticmethod
    def _output_nchw_to_rgb(output_nchw: TensorLike) -> TensorLike:
        output = output_nchw.squeeze(0).permute(1, 2, 0).contiguous()
        return output.mul_(255.0).clamp_(0, 255).byte()

    @staticmethod
    def _output_nchw_to_rgb_into(
        output_nchw: TensorLike,
        output_hwc: TensorLike,
        *,
        scratch_hwc: TensorLike | None = None,
    ) -> TensorLike:
        output_view = output_nchw.squeeze(0).permute(1, 2, 0)
        if scratch_hwc is None:
            output_hwc.copy_(output_view.mul(255.0).clamp_(0, 255))
            return output_hwc

        torch.mul(output_view, 255.0, out=scratch_hwc)
        torch.clamp(scratch_hwc, 0, 255, out=scratch_hwc)
        output_hwc.copy_(scratch_hwc)
        return output_hwc


TRTInference = TensorRTRuntime
