# Changes

This versioned changelog records notable user-facing and operational changes. It
is not a replacement for `git log` and does not list every patch or refactor.

Measured performance changes and benchmark comparisons are recorded separately
in `docs/PERFORMANCE_LOG.md`.

## Maintenance Rules

Add notable new changes to `Unreleased`, grouped by purpose:

```text
## Unreleased

### Fixed

- Corrected the production CV-CUDA color path for limited-range NV12. Y and UV
  code values are now expanded on the GPU before RGB inference and compressed
  before NVENC, while full-range input remains unchanged. Model-space capture
  continues to execute this shared production path.

### Added
### Changed
### Fixed
### Removed
```

List released versions in reverse chronological order.

`Unreleased` describes the net change from the latest released version, not
the implementation history within the release cycle. Fold intermediate fixes
and renames into the final feature description. Omit features that were both
introduced and removed before release; `git log` retains that history.

Include the following in `CHANGES.md`:

- CLI, Docker workflow, engine metadata, and runtime-default changes;
- output, encoding, color metadata, benchmark, or manifest behavior changes;
- project-structure migrations that affect agents or developers;
- breaking changes and manual migration steps.

Do not include:

- small refactor-only changes without behavioral impact;
- extraction of magic numbers into constants;
- typo or documentation cleanup that does not affect workflow;
- local runbook clarifications that do not change an interface or runtime
  behavior.

## Versioning

The version in `pyproject.toml` is increased for a release, not for every commit.

Before `1.0.0`, use pragmatic semantic versioning:

- `0.1.PATCH` - bug fixes, runbook/documentation fixes, and compatible
  operational changes;
- `0.MINOR.0` - a new feature, new CLI/workflow, or changed default behavior;
- `1.0.0` - when the CLI and Docker/runtime workflow are considered stable.

## Unreleased

### Added

- Licensed the project source and documentation under Apache License 2.0;
  third-party dependencies, models, and media retain their own licenses.
- Added a self-contained `make demo` GPU workflow with pinned
  RealESRGAN_x2plus weights, generated rich-media input, engine compilation,
  full output validation, and reusable verified intermediates. Model export
  also supports explicit `--size` and `--name` values.
- Added a media-preservation contract that stream-copies supported audio,
  subtitle, data, attachment, chapter, and metadata content. A preflight rejects
  incompatible output containers before inference, and successful processing
  publishes the result atomically.
- Added GitHub Actions checks for Ruff, mypy, compileall, unit and media
  integration tests, CLI smoke tests, benchmark-figure drift, and static
  Dockerfile validation. The lightweight checks image avoids downloading the
  TensorRT production base for regular CI.
- Added contribution and security policies, structured bug and feature issue
  forms, and complete package discovery metadata for the public repository.
- Added one goal-based benchmark interface for project regression, comparative,
  adaptive tuned, and diagnostic workflows. It prepares and verifies pinned
  assets, builds GPU-specific engines and images, runs smoke checks, records
  revision-bound resume state, and retains low-level Make targets for debugging.
- Added reproducible RealESRGAN_x2plus and SPAN x2 workloads at 720p and 1080p,
  pinned stock VSGAN and vs-mlrt environments, and diagnostic `trtexec`
  measurements.
- Added externally timed rotated campaigns with process-attributed CPU use,
  NVML power/utilization/VRAM sampling, lifecycle scopes, sanitized environment
  data, content hashes, output validation, and machine-checked workload
  contract versions.
- Added model-space tensor parity and full decoded-product PSNR/SSIM quality
  gates. Comparative results are publishable only when performance evidence and
  both quality contracts refer to the same assets, engines, images, and clean
  repository revision.
- Added upstream-default and adaptive best-tuned comparison profiles. Tuned
  selection uses short reconnaissance, full confirmation for the strongest
  candidates, explicit resource-limit evidence, and winner-only quality gates.
- Added reproducible Nsight Systems/NVTX diagnostics and generated light/dark
  benchmark figures. Privacy-reviewed RTX 3090 evidence and a validated compact
  JSON exporter are published under `benchmarks/results/`.

### Changed

- Renamed the project, Python distribution, import package, Docker images,
  production command, and benchmark implementation to `trtvideo`. Sources now
  use the standard `src/trtvideo/` layout, the CLI defaults to the
  `_processed` suffix, and no pre-release compatibility aliases are retained.
- The production runtime now has one GPU-resident contract:
  NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC. Runtime inference uses
  CV-CUDA-owned buffers and direct TensorRT bindings without importing PyTorch;
  PyTorch remains limited to model export.
- NVENC packets now stream into one long-lived FFmpeg mux process instead of a
  temporary elementary-stream file. MP4 `faststart`, stream preservation,
  shortened-output duration, and atomic publication are part of the same output
  contract.
- Replaced the single-subclass pipeline hierarchy with one explicit
  `NvcodecPipeline` orchestrator and typed `ProcessConfig`. Video metadata,
  FFprobe adaptation, SDR color policy, NVCodec frame processing, output
  preservation, muxing, profiling, and benchmark lifecycle recording now have
  separate module boundaries.
- Moved benchmark orchestration, CPU/NVML sampling, environment collection, and
  suite policy out of the production package and into the optional benchmark
  image. Benchmark Make targets live under `benchmarks/`; the root Makefile
  contains project build and quality checks.
- Comparative campaigns use explicit `execution_profile` terminology,
  immutable runner settings, append-only scheduling evidence, rotated order,
  three initial runs plus two when needed, and a four-of-five stability
  consensus after extension. Stage profiling remains diagnostic and is never
  used as end-to-end benchmark throughput.
- Reworked the root README to lead with exact tuned throughput, attributed CPU,
  peak VRAM, and a three-command validated demo before implementation details.
- Replaced the canonical animated Sintel benchmark input with a 70-second CC0
  live-action Madrid source. Preparation now records deterministic frame
  dropping by timestamp to 24 fps and restores ordinary x264 input B-frames,
  while the independent NVENC output contract remains B-frame-free. Legacy
  Sintel manifests and published snapshots remain immutable historical evidence.

### Fixed

- Corrected limited-range NV12 handling: video-range Y/UV values are expanded
  before RGB inference and compressed again before NVENC, while full-range input
  remains unchanged. Production and model-space capture share this exact
  CV-CUDA frame path.
- Kept NVDEC batches alive until their CUDA work completes and placed CV-CUDA,
  TensorRT, and NVENC on the intended stream, preventing asynchronous surface
  reuse. `--max-frames` also stops without fetching an unnecessary decoder
  batch before encoder flush.

### Removed

- Removed the CPU-frame/PyTorch runtime backend and the `--backend`, `--crf`,
  and product `--cuda-graph` options. Production video processing now exposes
  only the GPU-resident NVCodec contract.
- Removed `docker-compose.yml` and its duplicate fixed-path examples. The
  supported quick start is `make demo`; normal processing uses the explicit
  Docker commands in `README.md`.

## 0.3.1 - 2026-07-20

### Changed

- The non-profile `nvcodec` path now passes the runtime CUDA stream to NVENC and
  no longer synchronizes the host thread before every `Encode`, reducing CPU
  busy-wait without changing the CLI.
- Moved inference, TensorRT runtime, and backend documentation from agent
  instructions to public `docs/ARCHITECTURE.md`. `AGENTS.md` now contains only
  agent working rules and links to canonical documentation.

## 0.3.0 - 2026-07-12

### Changed

- Updated the base Docker image to `nvcr.io/nvidia/tensorrt:26.06-py3`.
- Migrated the TensorRT build workflow to TensorRT 11 strong typing. FP16 is now
  selected through `prepare-onnx --precision fp16`, and `build-engine` compiles
  the already typed ONNX without precision builder flags.
- Added `onnxconverter-common` to runtime/export dependencies for lightweight
  mixed-precision ONNX graph rewriting.

### Fixed

- The `nvcodec` backend now sets `gop` and `idrperiod` explicitly to
  approximately one keyframe per second, preventing one IDR/keyframe for the
  entire output.
- `prepare-onnx --precision fp16` no longer runs a ModelOpt/ONNX Runtime
  reference pass on a full frame and no longer requires more than 15 GB of
  memory for 1080p conversion.

### Removed

- Removed the runtime model registry and automatic engine discovery. `upscale`
  and `benchmark-upscale` now require explicit `--engine`; `--registry` was
  removed from `build-engine`. The `<engine>.json` sidecar remains metadata for
  one specific engine.
- Removed the obsolete `RUNBOOK_REALESRGAN_SPAN.md`.
- Removed the obsolete archived plan `docs/archive/TASKS.md`.
- Removed weak-typing `--fp16`, `--no-fp16`, and experimental `--fp16-io` flags
  from `build-engine`. Use an FP16 ONNX with TensorRT 11.

## 0.2.0 - 2026-05-31

### Added

- Added a `Makefile` with Docker-only `build-dev`, `check`, `test-unit`, `lint`,
  `typecheck`, and `compile` commands.
- Added a Docker-only pytest unit-test architecture for pure-Python contracts.
- The `nvcodec` backend now estimates target bitrate from source video bitrate
  by default.
- Automatic bitrate formula:
  `source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`.

### Changed

- `--bitrate-mbps` remains an explicit override for reproducible runs.
- `--crf` is no longer supported by `nvcodec`. The backend uses automatic
  bitrate from source metadata or explicit `--bitrate-mbps`. If source bitrate
  is unavailable, `--bitrate-mbps` must be provided.

### Fixed

- The `nvcodec` backend disables B-frames in NVENC (`bf=0`) to avoid reordered
  timestamps and `non monotonically increasing dts` errors during FFmpeg MP4
  validation.
- The `nvcodec` backend no longer rounds fractional FPS to an integer before
  passing it to the PyNvVideoCodec encoder. Mux still uses the exact
  `ffprobe r_frame_rate`.

## 0.1.0 - 2026-05-27

### Changed

- Established the project and Python package as `trtvideo`.
- Selected a root package layout without an additional `src/` layer.
- Renamed the Docker virtual environment to `/opt/trtvideo`.
- Updated Docker image examples to `trtvideo:latest`.

### Docs

- Replaced `CLAUDE.md` with `AGENTS.md`.
- Moved `OPTIMIZATIONS.md` to `docs/PERFORMANCE_LOG.md`.
- Moved `TASKS.md` to `docs/archive/TASKS.md`.
- Added `docs/ROADMAP.md` as the concise current plan.
- Removed the obsolete `scripts/run_batch.sh`.
