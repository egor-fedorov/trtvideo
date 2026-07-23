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

- Benchmark manifests and campaign summaries now contain separate lifecycle
  scopes: startup through the first completed frame, steady state through the
  last frame, and finalize/encoder flush/mux through process exit.
- Added a reproducible Stage 0 benchmark contract for RealESRGAN_x2plus and
  Sintel, including Docker-first `make -C benchmarks prepare` and
  `make -C benchmarks verify` commands, verifiable source hashes, and media/ONNX
  validation.
- Moved `benchmark-upscale` to an external end-to-end timer, a 3+2 run suite,
  NVML sampling, sanitized run manifests, and automatic FFmpeg output
  validation.
- Added the canonical `make -C benchmarks run-ai-media` command for the
  RealESRGAN_x2plus/Sintel workload.
- Isolated the benchmark runner and `nvidia-ml-py` in the optional `benchmark`
  Docker target. The production image does not receive benchmark-only
  dependencies or scripts.
- Added pinned Docker environments and separate runners for diagnostic
  `trtexec`, TensorRT 11 `vs-mlrt/vstrt` parity, and stock
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

- Updated the canonical benchmark workloads to
  `realesrgan-x2plus-sintel-v3` and `liveaction-span-sintel-v2`. The manifests
  now include immutable model-space frame selection and acceptance thresholds.
  The shared H.264 input no longer uses B-frames, simplifying strict frame-count
  and timestamp validation. Added selective `prepare --force-clips` without
  rebuilding ONNX.
- Separated per-stage profiling from benchmarking. Stage timings are available
  only through `upscale --profile/--profile-json`; `benchmark-upscale` launches
  the normal unprofiled pipeline in separate warmup and measured processes.
- Divided the benchmark roadmap into technical parity, stock product
  comparison, and diagnostics. Quality parity remains a separate gate before a
  publishable result.
- Moved benchmark-specific Make targets to `benchmarks/Makefile`. The root
  `Makefile` now contains only project build and quality-gate targets.
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

- Renamed the project to `ai-media-enhancer`.
- Renamed the Python package from `upscaler` to `ai_media`.
- Selected a root package layout without an additional `src/` layer.
- Renamed the Docker virtual environment to `/opt/ai-media-enhancer`.
- Updated Docker image examples to `ai-media-enhancer:latest`.

### Docs

- Replaced `CLAUDE.md` with `AGENTS.md`.
- Moved `OPTIMIZATIONS.md` to `docs/PERFORMANCE_LOG.md`.
- Moved `TASKS.md` to `docs/archive/TASKS.md`.
- Added `docs/ROADMAP.md` as the concise current plan.
- Removed the obsolete `scripts/run_batch.sh`.
