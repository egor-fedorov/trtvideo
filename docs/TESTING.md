# Testing

Default checks run through the lightweight Docker checks image defined in
`docker/checks.Dockerfile`. The local host does not need TensorRT,
PyNvVideoCodec, CV-CUDA, or a compatible Python runtime.

## Layers

### Unit

Fast pure-Python tests without GPU or runtime dependencies:

```bash
make build-dev
make test-unit
```

Unit tests must not import TensorRT, CV-CUDA, or PyNvVideoCodec.

The production reporting contract is kept in this layer: tests cover versioned
completion/progress serialization, wall-window ETA calculations, timing-scope
separation, CLI mapping, and rejection of colliding output destinations without
importing the GPU runtime.

Tests are grouped by the subsystem under test:

```text
tests/unit/trtvideo/    # CLI, video helpers, and model/engine tooling
tests/unit/benchmarks/  # benchmark runners, manifests, validation, and campaigns
```

Both groups remain unit tests. Real GPU and performance runs live in
`benchmarks/`, not in `tests/`.

Host benchmark command smoke tests use Python `-S` with the checkout's `src/`
on `PYTHONPATH`, disabling installed site-packages so the checks image cannot
mask undeclared host dependencies.

### Media Integration

The checks image also runs a GPU-free FFmpeg integration test:

```bash
make test-media-integration
```

It creates a synthetic MKV with two audio tracks, a subtitle, chapters, global
metadata, and an attachment. The test applies the shared output mapping,
validates every preserved stream with `ffprobe`, and fully decodes video plus
both audio tracks. A separate case confirms that MP4 preflight rejects source
streams that cannot be copied into MP4.

### Export Integration

CPU tests against the real PyTorch, ONNXScript, and ONNX Runtime libraries run
separately from the lightweight checks image:

```bash
make build-typecheck-runtime
make test-export-integration
```

`tests/model_tools/` checks the pixel-unshuffle translation both in eager mode
and through the registered PyTorch ONNX exporter. Channel-distinct inputs must
match PyTorch exactly, and the exported graph must not use the incompatible
`SpaceToDepth` lowering. No GPU, model weights, or TensorRT engines are needed.
The same layer checks engine-build precision metadata against real ONNX
protobufs, including FP32 bindings with FP16 weights and non-tensor value info.
These tests are not part of `make check` or its media integration layer.

### CLI/Docker Smoke

The non-GPU checks image validates Docker entrypoints. `benchmark-trtvideo` is
copied into this image only for test parity with the benchmark target; it is not
installed as a production project script.

`make cli-smoke` (also part of `make check`) prints one `PASS` line per
successful command. On failure it prints the command, exit code, and captured
stdout/stderr, then stops. Run an individual command below to view its full help.

```bash
docker run --rm trtvideo:dev trtvideo --help
docker run --rm trtvideo:dev trtvideo doctor --help
docker run --rm trtvideo:dev trtvideo compatibility-check --help
docker run --rm trtvideo:dev trtvideo compatibility-report --help
docker run --rm trtvideo:dev benchmark-trtvideo --help
docker run --rm trtvideo:dev export-onnx --help
docker run --rm trtvideo:dev prepare-onnx --help
docker run --rm trtvideo:dev prepare-compatibility-input --help
docker run --rm trtvideo:dev build-engine --help
```

Unit coverage for `compatibility-check` verifies checkpoint, static ONNX,
dynamic ONNX, custom-input, scale-error, dry-run, heartbeat, partial-cleanup,
context-mismatch, and artifact-invalidation paths without importing GPU runtime
libraries. The media integration layer exercises the shared FFmpeg fixture
command, so changes cannot silently diverge from `make demo`.

### GPU Smoke

The self-contained manual GPU smoke test is:

```bash
make demo
```

It builds the local `model-tools` target, then uses pinned RealESRGAN
weights and a verified five-second, 120-frame excerpt of the CC BY-SA 4.0
`Jacqueville beach in may 2026 (0)` live-action source. The prepared input and
validated result use browser- and desktop-friendly H.264/AAC MP4 containers.
The workflow covers model export, FP16 conversion, TensorRT engine build,
NVDEC/CV-CUDA/TensorRT/NVENC processing, mux, full decode, frame/timestamp/color
validation, relative input/output chroma-retention checking, and preservation
of source audio with an audible-signal threshold. Rich-container preservation
remains covered independently by the media integration tests. Model export
first compares a small
deterministic source-model probe between FP32 PyTorch and ONNX Runtime CPU and
retains the evidence beside the ONNX. Generated artifacts are cached in
`.demo/`; the prepared input sidecar binds its source hash and complete FFmpeg
command. Use `DEMO_FORCE=1` to rebuild every cached artifact.

Validate that:

- the output file exists;
- the resolution matches the scale factor;
- duration and frame count are close to the expected values;
- `pix_fmt` and color tags are correct;
- frames are not empty and the video does not freeze on the first frame.

The GPU-free media integration test runs the exact demo preparation command
against a deterministic broadband-audio fixture. A source-relative SI-SDR
threshold guards the pinned audio transcode without adding comparative quality
analysis to the demo runtime. Engine build and inference still require a GPU
host.

Before releasing the compatibility workflow, run one complete checkpoint case
on a GPU host, interrupt it after a successful engine build, and confirm that
`--resume` reruns `doctor`, skips verified stages, and completes the smoke/report
steps. Also run an existing static ONNX case in the production image and a
dynamic ONNX case with `--scale` in model-tools. These are acceptance checks,
not regular CI jobs.

### Benchmark

The report-first layer writes suite/run JSON, child logs, and raw NVML samples
without hard-coded FPS thresholds. Thresholds may be introduced only after
baselines have been collected for a specific GPU, TensorRT version, model, and
resolution.

Canonical benchmark assets are prepared and verified without a GPU:

```bash
make -C benchmarks prepare
make -C benchmarks verify
```

Preparation runs export conformance once per checkpoint, before engine builds.
The cached evidence is bound to the inferred scale, source SHA256, exporter
contract, tool versions, and generated FP32 ONNX hashes; verification rejects a
missing or stale report. It never runs inside a timed suite, tuned candidate, or
campaign round.

`prepare` downloads large ignored assets and is therefore excluded from the
regular quality gate. Pure-Python workload-manifest and preparation-command
contracts are covered by unit tests.

On a GPU host, first run a short runner and validation smoke test:

```bash
make -C benchmarks build
make -C benchmarks run-project \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  ARGS="--runs 1 --extra-runs 0 --frames 120 --warmup-frames 24 --idle-seconds 0"
```

Competitor Docker images, dry runs, and GPU acceptance commands are documented
in `benchmarks/GPU_RUNBOOK.md`. The normal GPU entrypoint is
`benchmarks/bin/run-benchmark.sh <goal>`; unit tests validate complete matrix
planning and revision-bound resume without a GPU. Make targets remain the
low-level test and troubleshooting interface. `project` is a project-only
regression measurement; only the rotated `comparative` or validated `tuned`
workflow can produce a competitor claim. A valid run must pass full decode,
media/timestamp validation, and NVML validity checks. `nvidia-ml-py` is
installed only in the optional `trtvideo:benchmark` image and is not
part of the production runtime.

The one-off `profile-nsight` diagnostic is also GPU-only, but it is not a
performance test: profiler overhead invalidates its FPS. Unit tests cover
command generation and opt-in NVTX behavior; GPU acceptance requires a valid
`.nsys-rep`, CLI stats reports, GPU video trace support, and a fully validated
120-frame output.

## Quality Gate

The minimum Docker gate for Python changes is:

```bash
make build-dev
make check
```

`make format` applies Ruff import sorting and Black-compatible formatting.
`make lint` checks both formatting and Ruff lint rules without changing files.
`make typecheck` runs both mypy and [ty](https://docs.astral.sh/ty/) over
`src/trtvideo` and `benchmarks`. Use `make typecheck-mypy` or
`make typecheck-ty` to run either checker separately in the development image.
ty is version-pinned in the dev dependencies; both tools are required gates,
not advisory checks. Both use the local [native API declarations](../stubs/README.md)
for CUDA bindings, TensorRT, CV-CUDA, and PyNvVideoCodec. ty allows unresolved
imports only for the remaining explicitly listed optional dependencies absent
from the lightweight checks image; missing project imports remain errors. Its
configuration does not reuse mypy's `type: ignore` suppressions.

`make typecheck-contracts` runs static positive and negative probes through both
checkers. The positive probes assert concrete types across the production frame
path; negative probes cover representative project/native boundaries, not every
SDK argument. Each intentionally invalid call must fail, not just produce a
nonzero exit code for the whole file. These tests live in `tests/typing/`, never
execute native libraries, and run as part of `make typecheck`, `make check`, the
full-dependency audit, and CI.

The runtime, production frame processor/pipeline, engine builder, and profiler
also enable stricter mypy rules: missing function annotations, bare generics,
explicit `Any`, and returning `Any` from typed functions are errors. The builder
keeps native imports lazy, so CLI help and pure-Python tests still work without
TensorRT. The profiler's event batches and summary fields retain their types;
quality capture keeps CUDA pointers, copy calls, and FP16/FP32 host arrays typed.
`make figures-check` regenerates benchmark SVGs in a temporary directory and
requires byte-for-byte equality with the committed figures.

`make check` does not rebuild the development image automatically. After
changes to dependencies in `pyproject.toml`/`uv.lock` or to
`docker/checks.Dockerfile`, run `make build-dev` first. A metadata-only project
version change does not require an image rebuild.

GitHub Actions builds the same checks image for pull requests and pushes to
`main`, then reports Ruff, mypy, ty, compileall, unit tests, media integration tests,
CLI smoke, and benchmark-figure drift as separate steps. A separate workflow
runs BuildKit static validation for the production Dockerfile without
downloading the 26 GB runtime image. Full production and benchmark builds remain
a GPU-host or self-hosted-runner acceptance check. Versioned releases build only
the production target, smoke-test `trtvideo` and `build-engine` by immutable
digest, and verify that model-conversion and benchmark commands are absent.

### Full-Dependency Type Audit

To run both type checkers with the installed runtime, export, and benchmark
libraries instead of the lightweight environment:

```bash
make build-typecheck-runtime
make typecheck-runtime
```

The build uses the root Dockerfile's `benchmark` target with `INSTALL_DEV=1`
and the same frozen dependency lock. It writes a separate
`trtvideo:typecheck-runtime` image without replacing production or benchmark
tags. Override `TYPECHECK_RUNTIME_IMAGE` on both commands to use another tag.
The checks read the current checkout; rebuild the image after Dockerfile,
base-image, or dependency changes. Neither the build nor the audit needs GPU
access. Runtime driver readiness is still checked separately with `trtvideo doctor`.

The audit first requires the full dependency set to be installed and prints its
versions. It then runs both checkers with the existing project configuration,
even if one fails, and preserves a nonzero exit status on failure. It is an
opt-in diagnostic, not part of `make check` or the lightweight CI gate.

The native libraries do not supply the needed static declarations. The local
stubs describe only the API used by the production pipeline, not the complete
vendor SDKs. Core tensors, streams, CUDA event results, decoder frames, and
encoded bitstreams retain their types through the frame path. TensorRT network,
parser, builder config, optimization profiles, timing cache, and serialized
buffers are covered as well. Native callables are bound directly where needed,
instead of passing modules through attributes that mypy treats as untyped
`ModuleType` objects. Real typed dependencies such as PyTorch and ONNXScript
are still checked against their installed declarations, not replaced with
project stubs.

This is not complete strict typing of the repository: environment discovery in
`doctor`, export tooling, and some benchmark/JSON adapters still use `Any`. Nor
does a green audit prove that a stub matches an updated native binary: check the
declarations against the installed API whenever those dependencies change.
Shape, dtype, color, ownership, and synchronization correctness still require
unit and GPU tests; static typing does not prove them.

The ONNXScript translator uses narrowly scoped `INT64` casts for automatically
promoted index literals and integer constants whose dtype is absent from the
generic operator signature. Casts do not change execution or graph operations;
`make test-export-integration` checks the real eager and export paths on CPU.
