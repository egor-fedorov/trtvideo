from collections.abc import Buffer, Iterator, Sequence
from enum import Enum
from typing import ClassVar, SupportsInt

__version__: str

class DataType(Enum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    UINT8 = 5
    FP8 = 6
    BF16 = 7
    INT64 = 8
    INT4 = 9
    FP4 = 10
    E8M0 = 11

float32: DataType

class TensorIOMode(Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2

class Logger:
    class Severity(Enum):
        INTERNAL_ERROR = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        VERBOSE = 4

    INTERNAL_ERROR: ClassVar[Severity]
    ERROR: ClassVar[Severity]
    WARNING: ClassVar[Severity]
    INFO: ClassVar[Severity]
    VERBOSE: ClassVar[Severity]
    min_severity: Severity
    def __init__(self, min_severity: Severity = ...) -> None: ...

class Dims:
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...

class Runtime:
    def __init__(self, logger: Logger) -> None: ...
    def deserialize_cuda_engine(self, serialized_engine: Buffer) -> ICudaEngine | None: ...

class ICudaEngine:
    @property
    def num_io_tensors(self) -> int: ...
    def create_execution_context(self) -> IExecutionContext | None: ...
    def get_tensor_name(self, index: SupportsInt) -> str: ...
    def get_tensor_shape(self, name: str) -> Dims: ...
    def get_tensor_dtype(self, name: str) -> DataType: ...
    def get_tensor_mode(self, name: str) -> TensorIOMode: ...

class IExecutionContext:
    def set_tensor_address(self, name: str, memory: SupportsInt) -> bool: ...
    def execute_async_v3(self, stream_handle: SupportsInt) -> bool: ...

class MemoryPoolType(Enum):
    WORKSPACE = 0

class IHostMemory:
    def __buffer__(self, flags: int, /) -> memoryview: ...

class ITensor:
    name: str
    shape: Dims
    dtype: DataType

class INetworkDefinition:
    def get_input(self, index: SupportsInt) -> ITensor | None: ...
    def get_output(self, index: SupportsInt) -> ITensor | None: ...

class ParserError: ...

class OnnxParser:
    def __init__(self, network: INetworkDefinition, logger: Logger) -> None: ...
    @property
    def num_errors(self) -> int: ...
    def parse(self, model: Buffer, path: str | None = None) -> bool: ...
    def get_error(self, index: SupportsInt) -> ParserError | None: ...

class ITimingCache:
    def serialize(self) -> IHostMemory | None: ...

class IOptimizationProfile:
    def set_shape(
        self,
        input: str,
        min: Dims | Sequence[int],
        opt: Dims | Sequence[int],
        max: Dims | Sequence[int],
    ) -> None: ...

class IBuilderConfig:
    builder_optimization_level: int
    def create_timing_cache(self, serialized_timing_cache: Buffer) -> ITimingCache | None: ...
    def set_timing_cache(self, cache: ITimingCache, ignore_mismatch: bool) -> bool: ...
    def get_timing_cache(self) -> ITimingCache | None: ...
    def set_memory_pool_limit(self, pool: MemoryPoolType, pool_size: SupportsInt) -> None: ...
    def add_optimization_profile(self, profile: IOptimizationProfile) -> int: ...

class Builder:
    def __init__(self, logger: Logger) -> None: ...
    def create_network(self, flags: SupportsInt = 0) -> INetworkDefinition | None: ...
    def create_builder_config(self) -> IBuilderConfig | None: ...
    def create_optimization_profile(self) -> IOptimizationProfile | None: ...
    def build_serialized_network(
        self, network: INetworkDefinition, config: IBuilderConfig
    ) -> IHostMemory | None: ...
