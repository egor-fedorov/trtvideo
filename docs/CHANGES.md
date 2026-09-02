# Changes

This versioned changelog records notable user-facing and operational changes. It
is not a replacement for `git log` and does not list every patch or refactor.

Measured performance changes and benchmark comparisons are recorded separately
in `docs/PERFORMANCE_LOG.md`.

## Maintenance Rules

Add notable new changes to `Unreleased`, grouped by purpose:

```text
## Unreleased

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

- Added `trtvideo compatibility-report` to verify and combine source identity,
  export conformance, static engine metadata, runtime readiness, exact commands,
  and complete smoke-output media checks into sanitized JSON and issue-ready
  Markdown.
- Added versioned production process reporting: `--result-json` writes one
  completion document and `--progress-jsonl` writes interval progress events,
  with `-` available as an exclusive machine-readable `stdout` destination.

### Changed

- Benchmark summary figures now show normalized end-to-end throughput with
  absolute FPS labels alongside linear CPU and peak VRAM comparisons.
- Documented where DeepStream, Video2X, chaiNNer, NVIDIA-accelerated FFmpeg, and
  VapourSynth fit relative to the project's deliberately narrow runtime scope;
  corrected the current reason Video2X remains outside the benchmark matrix.
- Defined maintainer-owned triage for model compatibility reports: reporters
  submit one issue with attachable generated evidence, accepted reports receive
  `community-reported`, and the maintainer owns the matrix pull request.
- Human process logs now use `stderr`; progress reports percentage, wall-window
  FPS, and frame-loop ETA, while the final summary separates frame-loop, active-
  pipeline, and in-process wall throughput. `--quiet` suppresses human progress
  and summary output without disabling requested machine reports.

## 0.5.0 - 2026-08-19

### Added

- Added a model compatibility matrix, a documented tensor/export contract, and
  a structured community compatibility report for models outside the published
  benchmark set.
- Published a separate privacy-reviewed RTX 4090 tuned and diagnostic snapshot.
  Benchmark figure generation and validation now discover every published
  hardware directory instead of checking only the RTX 3090 result.
- Added cached source-model export conformance before engine builds. A
  deterministic FP32 probe now compares the original PyTorch checkpoint with
  ONNX Runtime CPU under strict numerical thresholds; benchmark asset
  locks bind the evidence to checkpoint, toolchain, exporter contract, and
  generated FP32 ONNX hashes without adding work to timed campaigns.
- Added `trtvideo doctor`, a fast static environment readiness check for
  Docker execution, the NVIDIA driver and selected GPU, CUDA, TensorRT,
  CV-CUDA, NVDEC/NVENC, PyNvVideoCodec, VRAM, and writable disk capacity.

### Changed

- Added a bounded cross-GPU SPAN scaling analysis to the published RTX 4090
  result. The performance log now keeps current-runtime measurements in detail
  and condenses superseded legacy experiments into a retired summary.
- Replaced the synthetic one-second demo video with a pinned five-second
  CC BY-SA 4.0 beach excerpt with audible surf. The demo verifies and records
  provenance, attribution, modifications, source audio, and relative chroma;
  browser-friendly MP4 replaces the former rich MKV fixture, whose synthetic
  coverage remains isolated in integration tests. Prepared input caches are now
  bound to the source hash and complete FFmpeg command. High-bitrate AAC and
  a source-relative integration-test gate prevent audible degradation during
  preparation without adding comparative analysis to the demo runtime.
- Documented the short-lived topic-branch policy; merged pull-request branches
  are deleted automatically by the repository.
- Generalized `export-onnx` from a hard-coded 2x contract to a scale inferred
  from the source model. Versioned ONNX metadata and export-conformance evidence
  now bind the inferred scale and reject full-size exports that change it.

### Fixed

- Made resumed tuned workflows start untouched winner campaigns normally while
  retaining strict resume checks for campaigns that already have immutable
  evidence. Benchmark environment manifests now query the CUDA runtime directly
  through CUDA bindings instead of relying on `torch.version.cuda`, which is
  unavailable with the CPU-only model-export toolchain.
- Preserved PyTorch channel ordering when exporting x2 RRDB models that use
  `pixel_unshuffle`. The ONNX exporter previously emitted an incompatible
  `SpaceToDepth` operation, which mixed RGB channels and desaturated
  RealESRGAN_x2plus output. Versioned export metadata now invalidates affected
  cached demo models and engines automatically, benchmark asset verification
  rejects legacy graphs, and demo validation now checks that the deterministic
  color fixture retains a broad chroma range.
- Removed the implicit Python 3.11 requirement from release validation and
  Docker image builds. The release workflow uses POSIX `awk`, while Makefiles
  read the controlled project version with POSIX `sed`, preserving the
  benchmark orchestrator's Python 3.10 contract and failing before a build if
  the version is unavailable.

## 0.4.0 - 2026-08-14

### Added

- Licensed the project source and documentation under Apache License 2.0;
  added a redistribution inventory and third-party notices for dependencies,
  models, and media. A release-only GHCR workflow publishes the production
  target with a pinned base, SBOM, provenance, signed attestation, OCI metadata,
  and an immutable digest attached to the GitHub release.
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
  Dockerfile validation without downloading the TensorRT production base for
  regular CI.
- Added contribution and security policies, structured bug and feature issue
  forms, and complete package discovery metadata for the public repository.
- Added one goal-based benchmark interface for project regression, comparative,
  adaptive tuned, and diagnostic workflows, with reproducible RealESRGAN_x2plus
  and SPAN x2 workloads, pinned external environments, verified assets,
  GPU-specific engines, smoke checks, and revision-bound resume state.
- Added externally timed rotated campaigns with process-attributed CPU use,
  NVML power/utilization/VRAM sampling, lifecycle scopes, sanitized environment
  data, content hashes, output validation, and machine-checked workload
  contract versions.
- Added shared-input TensorRT parity, production-preprocessing diagnostics, and
  full decoded-product PSNR/SSIM evidence. Comparative results are publishable
  only when performance and quality evidence share the same assets, engines,
  images, contracts, clean repository revision, and recorded hardware, driver,
  and power-limit environment. Capture evidence fails closed on suspected
  cross-implementation output reuse.
- Added upstream-default and adaptive best-tuned comparison profiles. Tuned
  selection uses short reconnaissance, full confirmation for the strongest
  candidates, explicit resource-limit evidence, and winner-only quality gates.
- Added reproducible Nsight Systems/NVTX diagnostics and generated light/dark
  benchmark figures. Privacy-reviewed RTX 3090 evidence and validated compact
  tuned/diagnostic JSON exporters are published under `benchmarks/results/`.

### Changed

- Renamed the project, Python distribution, import package, Docker images,
  production command, and benchmark implementation to `trtvideo`. Sources now
  use the standard `src/trtvideo/` layout, the CLI defaults to the
  `_processed` suffix, and no pre-release compatibility aliases are retained.
- The production runtime now has one GPU-resident contract:
  NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC. Runtime inference uses
  CV-CUDA-owned buffers and direct TensorRT bindings without importing PyTorch;
  PyTorch remains limited to model export.
- Split local model export and ONNX conversion into the non-published
  `model-tools` image. The production image now contains only processing and
  engine-build commands; model-tools use CPU-only PyTorch, and benchmark builds
  inherit that toolchain without requiring a separate production build.
- NVENC packets now stream into one long-lived FFmpeg mux process instead of a
  temporary elementary-stream file. MP4 `faststart`, stream preservation,
  shortened-output duration, and atomic publication are part of the same output
  contract.
- Replaced the single-subclass pipeline hierarchy with one explicit
  `NvcodecPipeline` orchestrator and typed `ProcessConfig`, with separate media,
  NVCodec, muxing, profiling, and benchmark instrumentation boundaries.
- Moved benchmark orchestration, CPU/NVML sampling, environment collection, and
  suite policy out of the production package and into the optional benchmark
  image. Benchmark Make targets live under `benchmarks/`; the root Makefile
  contains project build and quality checks.
- Comparative campaigns use explicit `execution_profile` terminology,
  immutable settings, rotated order, adaptive stability checks, and
  machine-readable scheduling evidence. Stage profiling remains diagnostic and
  is never reported as end-to-end throughput.
- The canonical benchmark input is a 70-second CC0 live-action Madrid source.
  Preparation records deterministic frame dropping by timestamp to 24 fps and
  ordinary x264 input B-frames, while the independent NVENC output contract
  remains B-frame-free.

### Fixed

- Corrected limited-range NV12 handling: video-range Y/UV values are expanded
  before RGB inference and compressed again before NVENC, while full-range input
  remains unchanged. Production and the trtvideo tensor reference capture share
  this exact CV-CUDA frame path.
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

- Renamed the project to `ai-media-enhancer` and the Python package to
  `ai_media`.
- Selected a root package layout without an additional `src/` layer.
- Renamed the Docker virtual environment to `/opt/ai-media-enhancer`.
- Updated Docker image examples to `ai-media-enhancer:latest`.

### Docs

- Replaced `CLAUDE.md` with `AGENTS.md`.
- Moved `OPTIMIZATIONS.md` to `docs/PERFORMANCE_LOG.md`.
- Moved `TASKS.md` to `docs/archive/TASKS.md`.
- Added `docs/ROADMAP.md` as the concise current plan.
- Removed the obsolete `scripts/run_batch.sh`.
