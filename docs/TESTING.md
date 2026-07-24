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
tests/unit/ai_media/    # CLI, video helpers, and model/engine tooling
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

The non-GPU checks image validates Docker entrypoints:

```bash
docker run --rm ai-media-enhancer:dev upscale --help
docker run --rm ai-media-enhancer:dev benchmark-upscale --help
docker run --rm ai-media-enhancer:dev export-onnx --help
docker run --rm ai-media-enhancer:dev prepare-onnx --help
docker run --rm ai-media-enhancer:dev build-engine --help
```

### GPU Smoke

A future explicit GPU-host layer should use a short synthetic video and a tiny
TensorRT engine instead of real SPAN or RealESRGAN artifacts.

Validate that:

- the output file exists;
- the resolution matches the scale factor;
- duration and frame count are close to the expected values;
- `pix_fmt` and color tags are correct;
- frames are not empty and the video does not freeze on the first frame.

### Benchmark

The report-first layer writes suite/run JSON, child logs, and raw NVML samples
without hard-coded FPS thresholds. Thresholds may be introduced only after
baselines have been collected for a specific GPU, TensorRT version, backend,
model, and resolution.

Canonical benchmark assets are prepared and verified without a GPU:

```bash
make -C benchmarks prepare
make -C benchmarks verify
```

`prepare` downloads large ignored assets and is therefore excluded from the
regular quality gate. Pure-Python workload-manifest and preparation-command
contracts are covered by unit tests.

On a GPU host, first run a short runner and validation smoke test:

```bash
make -C benchmarks build
make -C benchmarks run-ai-media \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  ARGS="--runs 1 --extra-runs 0 --frames 120 --warmup-frames 24 --idle-seconds 0"
```

Competitor Docker images, dry runs, and GPU acceptance commands are documented
in `benchmarks/GPU_RUNBOOK.md`. The full 3+2 benchmark belongs to Stage 3. A
valid run must pass full decode, media/timestamp validation, and NVML validity
checks. `nvidia-ml-py` is installed only in the optional
`ai-media-enhancer:benchmark` image and is not part of the production runtime.

## Quality Gate

The minimum Docker gate for Python changes is:

```bash
make build-dev
make check
```

`make check` does not rebuild the development image automatically. After
changes to dependencies in `pyproject.toml`/`uv.lock` or to
`docker/checks.Dockerfile`, run `make build-dev` first. A metadata-only project
version change does not require an image rebuild.

GitHub Actions builds the same checks image for pull requests and pushes to
`main`, then reports Ruff, mypy, compileall, unit tests, media integration tests,
and CLI smoke as separate steps. A separate workflow runs BuildKit static
validation for the production Dockerfile without downloading the 26 GB runtime
image. Full production and benchmark builds remain a GPU-host or
self-hosted-runner acceptance check.
