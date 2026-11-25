"""Exercise runtime error paths with fake modules, without native GPU imports."""

import importlib.util
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_source(relative_path: str):
    spec = importlib.util.spec_from_file_location("runtime_under_test", ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCudaError(IntEnum):
    cudaSuccess = 0
    cudaErrorUnknown = 999


@pytest.fixture
def cuda_adapter(monkeypatch):
    runtime = ModuleType("cuda.bindings.runtime")
    runtime.cudaError_t = FakeCudaError
    runtime.cudaSetDevice = lambda device: (FakeCudaError.cudaSuccess,)
    runtime.cudaGetDeviceProperties = lambda device: (
        FakeCudaError.cudaSuccess,
        SimpleNamespace(name=b"Mock GPU\x00padding"),
    )
    runtime.cudaEventCreate = lambda: (FakeCudaError.cudaSuccess, 42)
    runtime.cudaEventRecord = lambda event, stream: (FakeCudaError.cudaSuccess,)
    runtime.cudaEventElapsedTime = lambda start, end: (FakeCudaError.cudaSuccess, 1.25)
    runtime.cudaEventDestroy = lambda event: (FakeCudaError.cudaSuccess,)
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.bindings = bindings
    bindings.runtime = runtime
    for module in (cuda, bindings, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return load_source("src/trtvideo/runtime/cuda.py"), runtime


def test_checked_cuda_results_preserve_values_and_report_errors(cuda_adapter) -> None:
    adapter, _ = cuda_adapter
    success = FakeCudaError.cudaSuccess
    token = object()
    assert adapter._checked((success, token, 1.25), "test") == (token, 1.25)
    assert adapter._checked((success,), "test") == ()
    assert adapter._checked(success, "test") == ()
    with pytest.raises(RuntimeError, match="test failed"):
        adapter._checked((FakeCudaError.cudaErrorUnknown,), "test")
    adapter.set_device(0)
    assert adapter.device_name(0) == "Mock GPU"


def test_cuda_events_preserve_stream_interop_and_close_once(cuda_adapter, monkeypatch) -> None:
    adapter, native = cuda_adapter
    records = []
    destroyed = []

    def record(event, stream):
        records.append((event, stream))
        return (FakeCudaError.cudaSuccess,)

    def destroy(event):
        destroyed.append(event)
        return (FakeCudaError.cudaSuccess,)

    monkeypatch.setattr(native, "cudaEventRecord", record)
    monkeypatch.setattr(native, "cudaEventDestroy", destroy)
    event = adapter.CudaEvent()
    event.record(1)
    event.record(SimpleNamespace(handle=2))
    event.record(SimpleNamespace(cuda_stream=3))
    assert records == [(42, 1), (42, 2), (42, 3)]
    assert event.elapsed_time(event) == 1.25
    with pytest.raises(TypeError, match="Unsupported CUDA stream"):
        event.record(object())
    event.close()
    event.close()
    assert destroyed == [42]


def test_failed_event_creation_does_not_destroy_an_uninitialized_event(
    cuda_adapter,
    monkeypatch,
) -> None:
    adapter, native = cuda_adapter
    destroyed = []
    monkeypatch.setattr(
        native,
        "cudaEventCreate",
        lambda: (FakeCudaError.cudaErrorUnknown, 0),
    )
    monkeypatch.setattr(native, "cudaEventDestroy", lambda event: destroyed.append(event))
    with pytest.raises(RuntimeError, match="cudaEventCreate failed"):
        adapter.CudaEvent()
    assert destroyed == []


@pytest.mark.parametrize("missing", ["engine", "context"])
def test_tensorrt_creation_failures_stop_before_tensor_allocation(
    monkeypatch,
    tmp_path: Path,
    missing: str,
) -> None:
    engine = None if missing == "engine" else SimpleNamespace(create_execution_context=lambda: None)
    fake_trt = SimpleNamespace(
        __version__="test",
        Logger=SimpleNamespace(WARNING=2),
        Runtime=lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda data: engine),
    )

    class Logger:
        WARNING = 2

        def __init__(self, severity):
            pass

    fake_trt.Logger = Logger
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "cvcuda", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "trtvideo.runtime.cuda",
        SimpleNamespace(
            CudaEvent=object,
            device_name=lambda device: "Mock GPU",
            set_device=lambda device: None,
        ),
    )
    module = load_source("src/trtvideo/runtime/cvcuda_tensorrt.py")
    path = tmp_path / "model.engine"
    path.write_bytes(b"fake engine")
    if missing == "engine":
        with pytest.raises(SystemExit) as error:
            module.CvcudaTensorRTRuntime(str(path))
        assert error.value.code == 1
    else:
        with pytest.raises(RuntimeError, match="Failed to create TensorRT execution context"):
            module.CvcudaTensorRTRuntime(str(path))
