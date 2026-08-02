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

- Added a benchmark publication exporter that converts a complete copied tuned
  session into the compact committed JSON while validating matrices, revision
  identity, evidence provenance, and external output fingerprints.
- Added machine-readable tuned-session power/temperature observations and an
  independent confirmation-to-final reproducibility control to distinguish
  cross-session GPU variation from within-session product comparisons.
- Added light/dark benchmark figures for tuned stream sweeps and direct
  trtvideo-versus-fastest-external CPU/VRAM comparisons at equivalent
  throughput. The SVGs are generated deterministically from the committed
  publication JSON and checked for drift in CI.
- Published post-rewrite RTX 3090 evidence with independently revisioned
  upstream-default, diagnostics, and adaptive best-tuned result classes for
  RealESRGAN and SPAN at 720p and 1080p. Both current tuned matrices and all
  winner quality gates are valid and publishable.
- Licensed the project source code and documentation under Apache License 2.0,
  with third-party dependencies, models, and media remaining under their own
  licenses.
- Added one goal-based benchmark command for complete `project`,
  `comparative`, `tuned`, and `diagnostics` workflows. A declarative matrix
  drives image and engine builds, asset preparation, smoke checks, quality
  gates, campaigns, diagnostics, dry-run plans, and revision-bound resume
  state; Make targets remain available for low-level troubleshooting.
- Added a manifest-driven tuned benchmark workflow with isolated candidate
  evidence, deterministic selection, winner-only quality gates, retained
  disqualifications, immediate phase progress, and machine-verified 720p plus
  1080p publication evidence.
- Added machine-checked benchmark contract versions to workload, run, and
  campaign manifests. Aggregation now rejects mixed frame-budget contracts.
- Split benchmark orchestration into explicit `run-project`,
  `run-comparative`, and diagnostic workflows with collision-free artifact
  namespaces. The GPU runbook now contains concrete 720p quality, campaign, and
  `trtexec` commands for both canonical models.
- Added a reproducible `make -C benchmarks profile-nsight` diagnostic for one
  SPAN 1080p `nvcodec` run. Opt-in NVTX ranges label the GPU pipeline, while the
  runner captures CUDA/NvVideo reports and validates the profiled output without
  treating profiler-affected FPS as a benchmark result.
- Added `make demo`, a self-contained GPU workflow using pinned and
  SHA256-verified RealESRGAN_x2plus weights plus a generated rich-media 720p
  input. It exports mixed-FP16 ONNX, builds the engine on the current GPU, runs
  `nvcodec`, and validates the complete 1440p output. Verified intermediates are
  cached under `.demo/`; `DEMO_FORCE=1` rebuilds generated assets.
- Added repeatable `export-onnx --size WIDTHxHEIGHT`; omitting it retains the
  existing 720p and 1080p defaults.
- Added a shared media-preservation contract and Docker FFmpeg integration
  tests. Upscaling now retains all source audio, subtitle, data, attachment,
  chapter, and metadata content when the selected output container supports it.
- Added an output-container preflight before TensorRT initialization.
  Incompatible stream-copy requests fail early and recommend MKV instead of
  dropping or transcoding source streams implicitly.
- Added GitHub Actions checks for Ruff, mypy, compileall, unit tests, CLI smoke,
  and BuildKit static validation of the production Dockerfile.
- Added a lightweight Python 3.12 checks image in `docker/checks.Dockerfile`.
  Regular quality checks no longer require downloading the NVIDIA/TensorRT base
  image.
- Benchmark manifests and campaign summaries now contain separate lifecycle
  scopes: startup through the first completed frame, steady state through the
  last frame, and finalize/encoder flush/mux through process exit.
- Project benchmark manifests now divide startup and finalize into internal
  lifecycle checkpoints for video probing, preservation preflight, runtime and
  codec setup, encoder flush, mux, cleanup, output commit, and process teardown.
  Suite and campaign summaries publish median checkpoint intervals without
  changing the external end-to-end timing contract.
- Added a reproducible Stage 0 benchmark contract for RealESRGAN_x2plus and
  Sintel, including Docker-first `make -C benchmarks prepare` and
  `make -C benchmarks verify` commands, verifiable source hashes, and media/ONNX
  validation.
- Moved `benchmark-upscale` to an external end-to-end timer, a 3+2 run suite,
  NVML sampling, sanitized run manifests, and automatic FFmpeg output
  validation.
- Added the canonical `make -C benchmarks run-trtvideo` command for the
  RealESRGAN_x2plus/Sintel workload.
- Isolated the benchmark runner and `nvidia-ml-py` in the optional `benchmark`
  Docker target. The production image does not receive benchmark-only
  dependencies or scripts.
- Added pinned Docker environments and separate runners for diagnostic
  `trtexec`, TensorRT 11 `vs-mlrt/vstrt`, and stock
  `VSGAN-tensorrt-docker` product comparison with a shared result schema, NVML
  sampling, output validation, and GPU-free `--dry-run`.
- Added a GPU benchmark runbook for a future acceptance campaign on RTX 3090.
- Added a second canonical workload using lightweight
  `2xLiveActionV1_SPAN`, including source hash, license/attribution, static ONNX
  variants, and shared Sintel clips.
- Added a separate TensorRT 10.16 engine build for stock VSGAN from the
  canonical ONNX, including build log, sidecar contract, and hashes. The
  project's TRT11 engine is not reused across incompatible runtime versions.
- Added an explicit `--name` to `export-onnx` so different supported Spandrel x2
  models can be exported reproducibly without a hard-coded RealESRGAN filename.
- Added `make -C benchmarks run-campaign`: implementations rotate by round,
  receive two additional runs automatically when spread exceeds 5%, and
  aggregate raw manifests into `campaign.json` plus sanitized `results.md`.
- Added process-attributed CPU accounting for the measured subprocess tree:
  user/system CPU seconds, average CPU cores, and affinity-normalized capacity.
  The rotated campaign publishes median CPU use and checks identical accounting
  semantics between products.
- Added a separate model-space parity gate for canonical frames `0`, `499`, and
  `999`. It captures normalized FP32 CHW RGB tensors before and after TensorRT
  from the project, TRT11 vstrt, and stock VSGAN, then produces a hashed JSON
  report using thresholds fixed before the GPU run.
- Added a separate product-output parity gate. It retains one canonical MP4 per
  implementation, compares all 1000 decoded frames with PSNR/SSIM, generates a
  fixed visual crop matrix, and hashes every report input. Stable campaigns
  become publishable only after both quality reports match their measured asset
  and engine contracts. Quality evidence also records immutable Docker image
  IDs and clean repository revision; aggregation reloads referenced run
  manifests and rejects evidence from another build.

### Changed

- Replaced the single-subclass `BasePipeline` template hierarchy with one
  explicit `NvcodecPipeline` orchestrator. CLI values now enter the runtime as
  an immutable `ProcessConfig`; color policy, atomic output publication,
  profiling reports, and benchmark lifecycle recording are separate cohesive
  collaborators. The NVDEC/CV-CUDA/TensorRT/NVENC frame order is unchanged.
- Separated normalized `VideoMetadata` from the FFprobe adapter. Probe failures
  now raise `VideoProbeError` instead of terminating the process, and the output
  subsystem is split into preservation, streaming mux, and atomic transaction
  modules behind the existing `trtvideo.video.output` import boundary.
- Renamed the production `upscale` entrypoint to `trtvideo`, its CLI module to
  `trtvideo.cli.process`, and the optional benchmark wrapper to
  `benchmark-trtvideo`. The default implicit output suffix is now `_processed`.
  This is a pre-release breaking change without compatibility aliases; the
  explicit model contract remains `task="upscale"` until another frame-local
  task is implemented.
- Replaced exhaustive tuned candidate suites with a two-stage adaptive search.
  One-run reconnaissance over streams `1..8` supports a validated decline stop,
  maximum-range sentinel, and unresolved-boundary rejection; only the three
  strongest points receive independent 1000-frame confirmation. Confirmation
  extends at 1% spread but rejects only beyond 5%, and the lowest-stream point
  within 1% of peak is selected to give competitors the most
  resource-efficient equivalent result. Short RealESRGAN reconnaissance records
  but does not enforce bitrate; all selection and publication evidence restores
  the full bitrate contract.
- Adaptive tuning now treats a machine-verified TensorRT/CUDA out-of-memory
  failure as a hashed resource ceiling instead of invalidating the full search.
  The selector independently revalidates the failed profile, stage policy,
  suite, run manifest, stderr path, and stderr SHA256; all other candidate
  failures remain fatal.
- Withdrew the pre-rewrite RTX 3090 benchmark snapshot. It predates the
  corrected limited-range color path and will be replaced only by evidence
  measured from one clean post-rewrite revision.
- Prepared the repository for public history by excluding local agent
  configuration, normalizing commit identity metadata, and removing legacy
  internal dependency references from early revisions.
- Quality-gate output now reports the current model-space and product-output
  operation out of four, while comparative campaigns retain their existing
  operation-level round counter.
- NVENC packets now stream directly into a long-lived FFmpeg mux process instead
  of being written to and reread from a temporary H.264/HEVC file. Stream
  preservation, MP4 `faststart`, and atomic output commit remain unchanged.
- The optional lightweight `nvtx` binding now provides Nsight ranges without
  importing PyTorch into the diagnostic process.
- Unified benchmark scheduling terminology around `execution_profile` across
  Make, runner CLIs, manifests, quality evidence, and campaigns. Comparative
  manifests no longer duplicate it as `comparison_class`, and individual
  implementation outputs use the `per-implementation` artifact namespace.
- Reorganized the benchmark harness around explicit contract, runtime, product,
  and VapourSynth adapter boundaries. Engine validation and process helpers are
  now shared without quality gates or external runners importing product-runner
  internals.
- Split the video package by abstraction level. Generic probe, FPS, frame
  iteration, and output-container contracts remain under `trtvideo.video`;
  NVDEC surface lifetime, CV-CUDA processing, bitrate policy, and NVENC settings
  now live under `trtvideo.video.nvcodec`. Ambiguous compatibility imports and
  legacy dict-style metadata access were removed.
- Replaced PyTorch tensor/stream orchestration in the default `nvcodec` runtime
  with CV-CUDA-owned buffers, CUDA Array Interface views, and direct TensorRT
  bindings. Ordinary GPU-resident inference no longer imports PyTorch; model
  export retains its PyTorch implementation.
- Expanded both tuned workload contracts with a mandatory VSGAN
  `num_streams=2..6` sweep using runtime-default VapourSynth threads. Contract
  validation now rejects an asymmetric grid.
- Standardized the project, Python distribution, import package, Docker images,
  build-provenance environment variables, benchmark implementation keys, and
  developer commands on the `trtvideo` name. The import package also moved from
  the root-level `trtvideo/` directory to the standard `src/trtvideo/` layout.
  This is a breaking migration without compatibility aliases; ignored campaign
  artifacts created with an older benchmark contract cannot be resumed.
- Moved benchmark process orchestration, CPU/NVML sampling, environment
  collection, and suite policy out of the production Python package. The
  `benchmark-upscale` wrapper now exists only in benchmark/check images, while
  campaign aggregation and tuned selection share one evidence-contract layer.
- Reduced the RealESRGAN warmup from 100 to 30 frames while retaining the
  1000-frame measured window. A trial 400-frame contract stayed below the
  startup-share limit but failed the fixed output-bitrate gate because the
  PyNvVideoCodec encoder cannot request NVENC filler-data insertion. SPAN and
  the full 1000-frame product-output quality gate are unchanged.

- Added validated `upstream-default` and `tuned` execution profiles
  to both VapourSynth benchmark runners. Automatic vspipe scheduling now omits
  `--requests`, while tuned runs require every scheduling choice explicitly.
- Isolated comparative and product-output artifacts by execution profile.
  Rotated campaigns now persist immutable runner settings and reject mixed
  scheduling contracts during resume, quality validation, and aggregation.
- Model-space captures now use and record the selected execution profile,
  including CUDA Graph, and are stored in profile-specific directories.
  Benchmark suites reject non-empty output directories to preserve sweep data.
- Comparative campaign stability now uses the full spread for three rounds and,
  after an automatic extension to five, accepts an explicit four-of-five
  consensus within the same 5% threshold. Raw values and headline medians still
  include all five runs; JSON and Markdown reports identify the accepted
  outlier and both spreads.
- Upscale output is now written to a same-directory temporary file and exposed
  atomically only after successful decode, encode, and mux. FFmpeg subprocess
  and final-mux failures return a non-zero status instead of leaving an
  apparently successful partial output. Chapters are omitted for
  `--max-frames` runs because original chapter timestamps may exceed the
  shortened output.
- Benchmark progress now identifies the project as `trtvideo` instead
  of exposing its selected `nvcodec` backend. `trtexec` artifact directories
  now include the workload name to prevent cross-model overwrites.
- Versioned the RealESRGAN and SPAN benchmark workloads after the first RTX 3090
  model-space run exposed an over-sensitive gate. Model-space acceptance now
  uses RMSE, p99, and PSNR; `max_abs` remains diagnostic, and input p99 allows
  `3/255` for differences between NVDEC/CV-CUDA and BestSource/zimg conversion.
- Updated the canonical benchmark workloads to
  `realesrgan-x2plus-sintel-v3` and `liveaction-span-sintel-v2`. The manifests
  now include immutable model-space frame selection and acceptance thresholds.
  The shared H.264 input no longer uses B-frames, simplifying strict frame-count
  and timestamp validation. Added selective `prepare --force-clips` without
  rebuilding ONNX.
- Separated per-stage profiling from benchmarking. Stage timings are available
  only through `upscale --profile/--profile-json`; `benchmark-upscale` launches
  the normal unprofiled pipeline in separate warmup and measured processes.
- Divided the benchmark roadmap into product comparison and diagnostics.
  Quality parity remains a separate gate before a publishable result.
- Moved benchmark-specific Make targets to `benchmarks/Makefile`. The root
  `Makefile` now contains only project build and quality-gate targets.
- Split benchmark tooling and its unit tests by responsibility: runners,
  campaign orchestration, quality gates, and workload preparation.
- Explicitly aligned the NVENC output contract for the project, vstrt, and
  VSGAN: single-pass CBR, equal target/min/max bitrate, a two-second VBV with
  50% initial occupancy, P4/HQ, disabled lookahead/AQ, one-second GOP, and zero
  B-frames.
- An individual benchmark suite now always has `scope: acceptance` and
  `publishable: false`. Only the rotated campaign determines comparative status,
  and it remains non-publishable until quality gates are complete.
- The rotated campaign now uses a Python coordinator and stores append-only
  `campaign.events.jsonl` with the actual order and pauses. The aggregator
  rejects untracked manifests and campaigns without a complete event log; all
  runners use the same 3+2 lifecycle and power-limit invariant.

### Fixed

- Propagated the effective bitrate-validation mode from external runner plans
  into VapourSynth suite and run manifests, so short tuning reconnaissance can
  prove that bitrate acceptance was intentionally disabled.
- Restored the model-space reference contract after the torch-free runtime
  migration. Production NVCodec inference and project tensor capture now share
  the same CV-CUDA frame processor instead of capturing through the retired
  PyTorch runtime path.
- Made Docker checks import the mounted `src/trtvideo` working tree instead of
  a potentially stale package copy installed when the development image was
  built.
- `--max-frames` no longer asks the NVDEC iterator for another decoder batch
  after the requested final frame. The iterator closes immediately and releases
  the current batch before encoder flush and mux.
- Short benchmark smoke runs now record actual bitrate without applying the
  full-campaign average-bitrate threshold. Canonical campaigns and
  product-output quality gates continue to enforce the fixed 10% tolerance.
- VSGAN benchmark scripts now treat an omitted `vs_threads` argument as the
  VapourSynth runtime default instead of failing before warmup.
- The comparative campaign coordinator now reads the generated campaign status
  instead of expecting GNU Make to preserve a sentinel exit code. Campaigns
  whose initial three rounds exceed the spread threshold now automatically run
  rounds four and five.
- The host-side benchmark campaign coordinator now supports the explicit
  Python `>=3.10,<3.13` range instead of failing on the Python 3.11-only
  `datetime.UTC` alias.
- The `nvcodec` pipeline now explicitly runs the NVDEC DLPack handoff,
  CV-CUDA color conversion, TensorRT, and NVENC on one CUDA stream. Decoder
  batches remain locked until that stream completes, preventing asynchronous
  surface reuse and stale-buffer reads while preserving batched NVDEC prefetch.
- Moved the stock VSGAN benchmark from the broken `minimal_no_avx512` image to
  pinned `latest_no_avx512` with native `vspipe`. The Docker build now checks
  the binary type and execution immediately. The benchmark virtual environment
  is no longer activated globally and no longer breaks embedded Python
  initialization in VSScript. The VSGAN engine sidecar records the base-image
  digest so an incompatible TensorRT plan is rejected before warmup. The
  external encoder is normalized to pinned Ubuntu FFmpeg 6.1.1 with `ffprobe`
  because upstream FFmpeg requires NVENC API 13.1 and driver 610+.
- SPAN ONNX export now folds mutating Spandrel `Conv3XC` blocks into equivalent
  evaluation convolutions once before `torch.export`. This resolves the PyTorch
  2.11 decomposition failure without switching to the deprecated legacy
  exporter.
- The `vstrt` runner now passes an absolute container path for input.
- Invalid benchmark runs now print concrete manifest errors to `stderr` before
  the Make target exits with code 2.
- Smoke overrides can no longer receive `publishable: true` accidentally. The
  suite summary checks exact equality with canonical workload parameters.
- The NVML process gate now accounts for the declared multi-process structure of
  external pipelines while retaining a zero baseline for detecting unrelated
  GPU load. Repeated NVML records for one PID no longer count as separate
  processes.

### Removed

- Removed the obsolete benchmark execution profile and its historical
  publication snapshot. Comparative runs now use pinned upstream defaults or
  a separately selected tuned configuration; model-space and product-output
  quality gates remain unchanged.
- Removed the CPU-frame video path, its PyTorch runtime adapter, and the
  `--backend`, `--crf`, and product `--cuda-graph` options. `upscale` now has one
  production contract: NVDEC, CV-CUDA, TensorRT, and NVENC, followed by FFmpeg
  stream-copy muxing.
- Removed `docker-compose.yml` and its duplicate fixed-path backend examples.
  The canonical quick-start path is now `make demo`; normal inference continues
  to use the explicit Docker commands in `README.md`.
- Removed Video2X from canonical benchmark tooling. Version 6.4.0 ran
  `realesr-animevideov3` instead of the required `RealESRGAN_x2plus`, so its FPS
  could not support a same-model performance claim.

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
