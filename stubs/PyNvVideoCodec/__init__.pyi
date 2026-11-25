from enum import Enum

from trtvideo.runtime.buffers import CudaArray, CudaBufferProvider

class OutputColorType(Enum):
    NATIVE = 0
    RGB = 1
    RGBP = 2

class DecodedFrame:
    def __dlpack__(self, stream: int | None = None) -> object: ...

class ThreadedDecoder:
    def __init__(
        self,
        enc_file_path: str,
        buffer_size: int,
        gpu_id: int = 0,
        cuda_context: int = 0,
        cuda_stream: int = 0,
        use_device_memory: bool = True,
        max_width: int = 0,
        max_height: int = 0,
        need_scanned_stream_metadata: int = 0,
        decoder_cache_size: int = 4,
        output_color_type: OutputColorType = ...,
        start_frame: int = 0,
        enableDecodeStats: bool = False,
    ) -> None: ...
    def get_batch_frames(self, batch_size: int) -> list[DecodedFrame]: ...
    def end(self) -> None: ...

class PyNvEncoder:
    def Encode(self, frame: CudaArray | CudaBufferProvider) -> bytes: ...
    def EndEncode(self) -> bytes: ...

def CreateEncoder(
    width: int, height: int, fmt: str, usecpuinputbuffer: bool, **kwargs: str | int
) -> PyNvEncoder: ...
