"""Run static probes, never import or execute GPU dependencies."""

import re
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = """\
from typing import assert_type
import cvcuda
import cuda.bindings.runtime as cudart
import PyNvVideoCodec as nvc
import tensorrt as trt
from trtvideo.runtime.cuda import CudaEvent, _checked
from trtvideo.cli.build_engine import load_timing_cache
from trtvideo.diagnostics.profiling import ProfileCollector, ProfileSummary
from benchmarks.scripts.quality.capture_trtvideo import _copy_gpu_tensor_to_host
from numpy.typing import NDArray
import numpy as np
from trtvideo.runtime.cvcuda_tensorrt import CvcudaTensorRTRuntime
from trtvideo.video.nvcodec.frame_processor import NvcodecFrameProcessor
from trtvideo.video.nvcodec.surface import nv12_plane_views

def probe(runtime: CvcudaTensorRTRuntime, processor: NvcodecFrameProcessor,
          decoder: nvc.ThreadedDecoder, encoder: nvc.PyNvEncoder,
          context: trt.IExecutionContext, frame: nvc.DecodedFrame,
          builder: trt.Builder, config: trt.IBuilderConfig, network: trt.INetworkDefinition,
          profile: trt.IOptimizationProfile, parser: trt.OnnxParser,
          cache: trt.ITimingCache, collector: ProfileCollector) -> None:
"""
VALID = """\
    tensor = runtime.gpu_input
    assert_type(tensor, cvcuda.Tensor)
    assert_type(tensor.cuda().__cuda_array_interface__["data"][0], int)
    assert_type(runtime.execute(), cvcuda.Tensor)
    assert_type(runtime.infer({runtime.input_name: tensor}), dict[str, cvcuda.Tensor])
    assert_type(processor.wrap_nv12(frame), cvcuda.Tensor)
    assert_type(processor.preprocess(tensor), cvcuda.Tensor)
    assert_type(processor.postprocess(), cvcuda.Tensor)
    # A stored ModuleType must not silently turn native callables into Any.
    assert_type(processor._convertto_into(tensor, tensor, stream=runtime.stream), cvcuda.Tensor)
    assert_type(processor.buffers.rgb_in_u8, cvcuda.Tensor)
    assert_type(decoder.get_batch_frames(1), list[nvc.DecodedFrame])
    assert_type(encoder.Encode(tensor), bytes)
    assert_type(context.execute_async_v3(runtime.stream_handle), bool)
    y, uv = nv12_plane_views(tensor, height=8, width=8)
    assert_type(cvcuda.as_tensor(y, layout="NHWC"), cvcuda.Tensor)
    assert_type(cvcuda.as_tensor(frame), cvcuda.Tensor)
    assert_type(_checked(cudart.cudaEventCreate(), "create"), tuple[cudart.cudaEvent_t])
    assert_type(_checked(cudart.cudaSetDevice(0), "set"), tuple[()])
    event = CudaEvent()
    event.record(runtime.stream)
    assert_type(event.elapsed_time(event), float)
    assert_type(builder.create_network(), trt.INetworkDefinition | None)
    assert_type(builder.build_serialized_network(network, config), trt.IHostMemory | None)
    assert_type(config.set_timing_cache(cache, False), bool)
    assert_type(config.add_optimization_profile(profile), int)
    profile.set_shape("input", (1, 3, 8, 8), (1, 3, 8, 8), (1, 3, 8, 8))
    assert_type(parser.parse(b"model"), bool)
    serialized_cache = cache.serialize()
    if serialized_cache is not None:
        assert_type(bytes(serialized_cache), bytes)
    assert_type(load_timing_cache(config, None), bool)
    collector.commit((event, event))
    assert_type(collector.summary([]), ProfileSummary)
    assert_type(_copy_gpu_tensor_to_host(tensor), NDArray[np.float16 | np.float32])
"""
# Representative boundaries, not an exhaustive test of each SDK argument.
INVALID = """\
    CudaEvent().record("stream")
    context.set_tensor_address("input", "pointer")
    decoder.get_batch_frames("eight")
    encoder.Encode(123)
    processor.preprocess(b"not a tensor")
    runtime.infer({"input": 123})
    nv12_plane_views(123, height=8, width=8)
    config.set_timing_cache(b"cache", False)
    profile.set_shape("input", (1, "three", 8, 8), (1, 3, 8, 8), (1, 3, 8, 8))
    network.get_input(0).name
    parser.parse("not a buffer")
    collector.commit((1, 2))
    _copy_gpu_tensor_to_host(123)
    processor._convertto_into(runtime.gpu_input, runtime.gpu_output, stream="not a stream")
"""


def check(checker: str, path: Path) -> subprocess.CompletedProcess[str]:
    if checker == "mypy":
        command = [
            "mypy",
            "--config-file",
            str(ROOT / "pyproject.toml"),
            "--no-incremental",
            "--follow-imports=silent",
            "--show-error-codes",
            str(path),
        ]
    else:
        command = ["ty", "check", "--project", str(ROOT), "--output-format", "concise", str(path)]
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("checker", ["mypy", "ty"])
def test_native_api_accepts_typed_production_calls(checker: str, tmp_path: Path) -> None:
    probe = tmp_path / "valid_native_api.py"
    probe.write_text(HEADER + VALID, encoding="utf-8")
    result = check(checker, probe)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("checker", ["mypy", "ty"])
def test_native_api_rejects_every_invalid_call(checker: str, tmp_path: Path) -> None:
    probe = tmp_path / "invalid_native_api.py"
    probe.write_text(HEADER + INVALID, encoding="utf-8")
    result = check(checker, probe)
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 1, diagnostics
    reported_lines = {int(line) for line in re.findall(rf"{probe.name}:(\d+):", diagnostics)}
    expected_lines = set(range(HEADER.count("\n") + 1, (HEADER + INVALID).count("\n") + 1))
    assert reported_lines == expected_lines, diagnostics
    assert "unresolved-import" not in diagnostics
    assert "import-not-found" not in diagnostics


def test_type_only_imports_do_not_require_gpu_libraries() -> None:
    # -S also disables installed native packages in the full-dependency image.
    result = subprocess.run(
        [
            "python3",
            "-S",
            "-c",
            dedent("""\
            import sys
            import trtvideo.runtime
            import trtvideo.runtime.buffers
            import trtvideo.video.nvcodec.surface
            import trtvideo.video.nvcodec.frame_processor
            import trtvideo.cli.build_engine
            import trtvideo.diagnostics.profiling
            import benchmarks.scripts.quality.capture_trtvideo
            assert not {"cvcuda", "tensorrt", "PyNvVideoCodec", "cuda"}.intersection(sys.modules)
        """),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
