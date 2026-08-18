# Testing

Project tests run through the lightweight Docker checks image defined in
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

Tests are grouped by the subsystem under test:

```text
tests/unit/trtvideo/    # CLI, video helpers, and model/engine tooling
tests/unit/benchmarks/  # benchmark runners, manifests, validation, and campaigns
```

Both groups remain unit tests. Real GPU and performance runs live in
`benchmarks/`, not in `tests/`.

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

### CLI/Docker Smoke

The non-GPU checks image validates Docker entrypoints. `benchmark-trtvideo` is
copied into this image only for test parity with the benchmark target; it is not
installed as a production project script.

```bash
docker run --rm trtvideo:dev trtvideo --help
docker run --rm trtvideo:dev trtvideo doctor --help
docker run --rm trtvideo:dev benchmark-trtvideo --help
docker run --rm trtvideo:dev export-onnx --help
docker run --rm trtvideo:dev prepare-onnx --help
docker run --rm trtvideo:dev build-engine --help
```

### GPU Smoke

The self-contained manual GPU smoke test is:

```bash
make demo
```

It builds the non-published `model-tools` target, then uses pinned RealESRGAN
weights and a verified five-second, 120-frame 720p excerpt of the CC0
`Madrid-2021-05-06` live-action source.
The workflow covers model export, FP16 conversion, TensorRT engine build,
NVDEC/CV-CUDA/TensorRT/NVENC processing, mux, full decode, frame/timestamp/color
validation, relative input/output chroma-retention checking, and preservation
of the source audio. Rich-container preservation remains covered independently
by the media integration tests. Model export first compares a small
deterministic source-model probe between FP32 PyTorch and ONNX Runtime CPU and
retains the evidence beside the ONNX. Generated artifacts are cached in
`.demo/`; use `DEMO_FORCE=1` to rebuild them.

Validate that:

- the output file exists;
- the resolution matches the scale factor;
- duration and frame count are close to the expected values;
- `pix_fmt` and color tags are correct;
- frames are not empty and the video does not freeze on the first frame.

The GPU-free media integration test runs the exact demo preparation command
against a local media fixture and validates its result. Engine build and
inference still require a GPU host.

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
`make figures-check` regenerates benchmark SVGs in a temporary directory and
requires byte-for-byte equality with the committed figures.

`make check` does not rebuild the development image automatically. After
changes to dependencies in `pyproject.toml`/`uv.lock` or to
`docker/checks.Dockerfile`, run `make build-dev` first. A metadata-only project
version change does not require an image rebuild.

GitHub Actions builds the same checks image for pull requests and pushes to
`main`, then reports Ruff, mypy, compileall, unit tests, media integration tests,
CLI smoke, and benchmark-figure drift as separate steps. A separate workflow
runs BuildKit static validation for the production Dockerfile without
downloading the 26 GB runtime image. Full production and benchmark builds remain
a GPU-host or self-hosted-runner acceptance check. Versioned releases build only
the production target, smoke-test `trtvideo` and `build-engine` by immutable
digest, and verify that model-conversion and benchmark commands are absent.
