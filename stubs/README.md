# Native API Type Declarations

These project-local `.pyi` files describe the subset of native GPU APIs used by
trtvideo's production frame path and engine builder. They are not replacement
Python modules or complete vendor SDK typings. Both mypy and ty resolve them
before installed untyped native packages, in the lightweight and full-dependency
checks images.
They are not installed in the production wheel or added to runtime `PYTHONPATH`.

## Sources

The baseline declarations were checked against the packages in
`trtvideo:typecheck-runtime`:

| Package | Version | Signature source |
|---|---|---|
| TensorRT | 11.2.1.2 | Installed Python API docstrings and [runtime API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/Runtime.html) |
| cuda-bindings | 13.2.0 | Installed `cuda.bindings.runtime` signatures/docstrings and [runtime bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html) |
| cvcuda-cu12 | 0.16.0 | Installed pybind signatures and [tensor API](https://cvcuda.github.io/CV-CUDA/_python_api/nvcv/tensor.html) |
| PyNvVideoCodec | 2.1.0 | Installed `ThreadedDecoder.py`, `CreateEncoder`, and encoder pybind signatures |

CV-CUDA and codec extensions cannot be imported without NVIDIA driver libraries.
Their embedded pybind signatures can also be inspected in the installed `.so`
files; this does not verify GPU execution. Prefer live API inspection and a short
GPU smoke test when a GPU host is available.

`trtvideo.runtime.buffers` defines the shared CUDA Array Interface and DLPack
structural types without importing native libraries. The buffer interfaces
follow the [CUDA Array Interface](https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html)
and [DLPack](https://data-apis.org/array-api/latest/design_topics/data_interchange.html)
contracts. The declarations cover the dense device buffers used here; for
example, masked arrays and empty-array null pointers are not modeled.

## Scope And Maintenance

- Declare only verified members and overloads that callers need. Unsupported
  members intentionally cause errors; there is no blanket `__getattr__ -> Any`.
- CUDA calls return tuples beginning with `cudaError_t`; the adapter preserves
  the types of the remaining tuple values rather than converting them to `Any`.
- TensorRT declarations cover inference and the engine builder: network/parser,
  builder config, optimization profiles, timing caches, and serialized buffers.
  Object creation can fail; callers must narrow `None` before accessing these
  objects. Shapes remain runtime-validated integers, not a statically proven NCHW
  layout. `set_shape` returns `None` or raises `ValueError`; adding a profile
  returns its index or `-1`. Do not assume these APIs all return booleans.
- CV-CUDA declarations cover tensor allocation, CUDA-buffer export, streams, and
  the three conversion operators used in the pipeline. They omit image batches,
  unused dtypes, and overloads not exercised by this project.
- Codec declarations cover the device-memory threaded decoder and GPU encoder
  input used here. Encoder option dictionaries remain dynamically validated by
  the vendor API; static typing of `**kwargs` does not validate option names.
- Dependency upgrades require reviewing signatures as well as rerunning
  `make typecheck-runtime`. A checker using these stubs cannot by itself detect
  a native API change. Remove local declarations if suitable vendor typings
  become available, rather than shadowing them indefinitely.

Run `make typecheck-contracts` to verify concrete inferred return types and
rejection of representative invalid calls at project/native boundaries. Keep
these probes focused on loss of type information (especially `Any`), buffer and
stream contracts, and nullable native results, rather than testing every SDK
argument. The same tests run in the full audit and CI without importing GPU
libraries. These tests guard typing coverage, not the numerical correctness or
binary compatibility of the native implementation.
