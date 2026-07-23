# AI Media Enhancer - Agent Guidelines

## Project Context

`ai-media-enhancer` is a Docker-first collection of CLI tools for TensorRT-based
AI media processing. The currently implemented workflow is video upscaling
through either the `ffmpeg` or `NVDEC/NVENC` backend.

The production runtime uses Python 3.12 from the
`nvcr.io/nvidia/tensorrt:26.06-py3` base TensorRT Docker image. Development is
performed locally, while checks that require TensorRT, PyNvVideoCodec, CV-CUDA,
or a GPU normally run in Docker on a remote GPU host.

## Sources Of Truth

- `README.md` - public Docker workflow, CLI, and model preparation.
- `docs/ARCHITECTURE.md` - inference, runtime, and backend architecture.
- `docs/TESTING.md` - test layers and the Docker-only quality gate.
- `docs/ROADMAP.md` - concise current plan.
- `docs/CHANGES.md` - notable changes and versioning rules.
- `docs/PERFORMANCE_LOG.md` - measured performance changes.
- `benchmarks/methodology.md` - reproducible benchmark contract.

Do not duplicate user-facing or architectural documentation in this file. When
behavior changes, update the corresponding canonical document.

## Working Rules

- The primary workflow is Docker-first. Do not treat missing runtime-only
  dependencies on the local host as a project error.
- Do not commit `models/`, `videos/`, or large runtime artifacts without an
  explicit request.
- Model weights, ONNX files, and TensorRT engines are not vendored in the
  repository.
- Run a short smoke test with `--max-frames` before a full batch run.
- For changes to the color or encoding path, verify more than process startup.
  Use `ffprobe` to check `pix_fmt`, `color_range`, `color_space`,
  `color_transfer`, `color_primaries`, bitrate, duration, frame count, and frame
  timestamps.
- Record notable workflow, CLI, Docker, file-structure, and project-policy
  changes in `docs/CHANGES.md`.
- Record performance changes in `docs/PERFORMANCE_LOG.md` only when a
  measurement is available: what changed, which benchmark was used, and the
  resulting improvement or regression.
- If a check cannot run locally because GPU/runtime dependencies are missing,
  state that explicitly in the final response.
- Unit tests must remain pure Python and must not import TensorRT, CV-CUDA, or
  PyNvVideoCodec.
- Do not invalidate the Docker dependency cache unnecessarily: dependency
  metadata is copied before application code, and the project is installed in a
  separate layer.

## Checks

Validation tools are installed in the Docker development image:

```bash
make build-dev
make check
```

After Python changes, run at least `ruff check .` through the development image.
Before committing Python code, run the complete `make check` gate: Ruff, mypy,
compileall, and unit tests.

GPU/runtime smoke tests and benchmarks run on a GPU host. Commands and acceptance
criteria are documented in `README.md` and `docs/TESTING.md`.
