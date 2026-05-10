"""TensorRT inference wrapper with GPU-side pre/postprocessing."""

import sys

import numpy as np
import tensorrt as trt
import torch


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
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
                f"  Engine may be built with a different TensorRT version (current: {trt.__version__})."
            )
            print("  Rebuild: build-engine <model>.onnx")
            sys.exit(1)

        self.context = self.engine.create_execution_context()

        # Determine shapes
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

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

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Upscale a single frame (CPU->GPU->TRT->GPU->CPU).

        Args:
            frame_rgb: Input frame (H, W, 3) uint8 RGB.

        Returns:
            Upscaled frame (H*scale, W*scale, 3) uint8 RGB.
        """
        with torch.cuda.stream(self.stream):
            frame_gpu = torch.from_numpy(frame_rgb).to(device=self.device, non_blocking=True)
            frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            self.gpu_input.copy_(frame_gpu)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            output = self.gpu_output.squeeze(0).permute(1, 2, 0)
            output = output.mul_(255.0).clamp_(0, 255).byte()

        self.stream.synchronize()
        return output.cpu().numpy()
