# Roadmap

The goal of the next cycle is to determine whether `ai-media-enhancer` has a
measurable advantage as an NVIDIA/TensorRT video upscaler and to prepare
defensible performance claims for an open-source release.

The architectural claim under test is:

> The GPU-resident `NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` pipeline
> provides high end-to-end throughput without transferring uncompressed frames
> to the CPU.

## Benchmark Matrix

Results are divided into independent classes:

1. Technical parity: `ai-media-enhancer` against locally built
   `VapourSynth/vstrt` on TensorRT 11 with the same serialized engine.
2. Stock product: `ai-media-enhancer` against pinned stock
   `VSGAN-tensorrt-docker` with the same ONNX but separate native engines. Stock
   VSGAN uses TensorRT 10.16 and cannot load a TensorRT 11 engine.
3. Diagnostics: `trtexec`, stage profiling, and Nsight. These are not
   competitors and do not appear as rows in the product table.

Video2X is excluded: version 6.4.0 does not support the universal
`RealESRGAN_x2plus` used here and runs a different anime model. Comparing its FPS
would not test pipeline efficiency.

Mandatory workloads:

- `RealESRGAN_x2plus` - a heavy, model-bound scenario;
- `2xLiveActionV1_SPAN` - a light scenario that exposes video-pipeline overhead.

## Stage 0. Rebaseline Tooling

Status: implemented offline. `make check`, benchmark/TRT11 vstrt image builds,
runner dry runs, and the static VSGAN Dockerfile check pass. The complete stock
VSGAN image build, TensorRT engines, and runtime smoke tests belong to GPU
acceptance in Stage 1.

- Remove Video2X from canonical tooling and documentation.
- Move `trtexec` to the diagnostic/reference class.
- Add a pinned stock VSGAN image without changing its internal code.
- Keep a strict TRT11 `vstrt` runner for technical parity.
- Add a SPAN workload with verifiable source hash, attribution, and license.
- Build a TRT10.16 VSGAN engine from the canonical mixed-FP16 ONNX on the
  benchmark GPU and retain the build log, engine sidecar, and hashes.
- Define success criteria and the complete inference/output contract in
  `benchmarks/methodology.md`.

## Stage 1. GPU Acceptance

The target card for the first campaign is one physical GeForce RTX 3090 with
24 GB of VRAM.

- Record driver, power limit, clocks, thermal state, and immutable image IDs.
- Build project TRT11 engines and stock VSGAN TRT10.16 engines for RealESRGAN and
  SPAN on this GPU. Never reuse engines across TensorRT runtimes.
- Run a 720p smoke test for every runner: project, TRT11 `vstrt`, stock VSGAN,
  and diagnostic `trtexec`. Repeat at 1080p.
- For each video output, validate complete decode, frame count, timestamps, color
  tags, GOP/B-frames, bitrate, and size.
- Any image, engine, model, or setting change invalidates the affected series
  and requires another smoke test.

## Stage 2. Measurement Gaps

Status: complete for the published RTX 3090 1080p baseline. The exact NVENC
contract, rotated campaign runner, sanitized acceptance table, CPU accounting,
lifecycle timings, and both quality gates passed. The 720p confirmation belongs
to Stage 3.

- [x] Explicit requested NVENC rate-control contract for the project and VSGAN:
  codec, preset, tuning, RC mode, target/min/max bitrate, VBV, GOP, and B-frames.
  Actual bitrate and implementation-specific strict-CBR filler are reported
  separately.
- [x] CPU utilization for the measured subprocess tree through
  `RUSAGE_CHILDREN`: user/system CPU seconds, average cores, and
  affinity-normalized capacity.
- [x] Separate `startup`, steady-state frame loop, and `finalize + mux` timing
  scopes with one process/frame boundary contract.
- [x] Model-space parity on RGB/float frames before YUV conversion and encode.
  Capture/compare tooling and fixed thresholds passed for both 1080p workloads
  on the RTX 3090.
- [x] Product-output PSNR/SSIM and visual crops after decoding MP4.
  Retained-output runs, full-decode metrics, crop generation, and aggregator
  validation passed for both 1080p workloads.
- [x] Campaign runner that rotates products by round instead of running grouped
  suites.
- [x] Sanitized final acceptance-table generation from raw manifests.

Individual runners remain acceptance/baseline data. Only an aggregated campaign
with completed quality gates is publishable.

## Stage 3. Parity Campaign

Status: the validated RTX 3090 `1080p -> 4K` campaigns for RealESRGAN and SPAN
remain a valid historical baseline for revision `49ae95a` and are published in
`benchmarks/results/rtx-3090/1080p/`. Media preservation changed the
full-process finalize/mux path after that revision, so a current-release 1080p
rebaseline and the `720p -> 1440p` confirmation remain pending.

- Run 100 warmup and 1000 measured frames, at least three runs, and two
  additional runs when spread exceeds 5%.
- Rotate project/vstrt/VSGAN order between rounds.
- Run `1080p -> 4K` first, followed by confirmation at `720p -> 1440p`.
- Repeat both resolutions for RealESRGAN and SPAN.
- Publish median end-to-end FPS, wall time, CPU, average power, joules/frame,
  peak VRAM, output bitrate, and size.
- Retain commands, environment, raw values, and hashes. Do not mix results from
  different commits, images, or thermal/power states.

The success criterion is fixed before measurement:

- more than a 5% median end-to-end FPS advantage is a confirmed speed advantage;
- a difference within +/-5% is parity, followed by comparison of CPU,
  energy/frame, VRAM, and UX;
- losing by more than 5% on both workloads requires profiling and optimization
  before making a performance claim.

## Stage 4. Diagnostics And Best-Tuned

Status: 1080p `trtexec` ceilings and pipeline efficiency are published. The
first Nsight trace confirmed a GPU-resident, compute-saturated steady state, but
TensorRT reported that its engine was built for a different device model. A
clean trace with an engine rebuilt on the profiling GPU remains pending,
together with best-tuned and live-action confirmation runs.

- [x] Calculate `pipeline efficiency = end-to-end FPS / trtexec QPS` separately
  from the product table for the 1080p baseline.
- Collect one representative Nsight Systems trace and inspect H2D/D2H copies,
  PCIe traffic, stream gaps, CPU waits, and NVDEC/TensorRT/NVENC overlap.
- Run a separate best-tuned benchmark: VSGAN with recommended requests, streams,
  and CUDA Graph; the project with its best verified settings.
- Keep CUDA Graph experimental while it captures only the TensorRT call and
  provides no measured benefit on current heavy models.
- Repeat the headline workload on a short live-action clip with substantial
  motion and fine detail.
- If the timeline confirms idle periods, investigate double buffering, multiple
  execution contexts, and overlap of `decode N+1`, `inference N`, and
  `encode N-1`.
- After each change, repeat one fixed benchmark and add only measured effects to
  `docs/PERFORMANCE_LOG.md`.

## Stage 5. Open-Source Release

- Add `LICENSE` and audit dependency, model, and media licenses.
- [x] Keep the English README as the primary README and add a one-command
  cached GPU demo covering model download through validated rich-media output.
- [x] Add CI for Ruff, mypy, pytest, and static Dockerfile validation without
  requiring a GPU.
- Move full production and benchmark Docker builds to a larger or self-hosted
  GitHub Actions runner with enough disk for the 17 GB TensorRT base and 26 GB
  final images. Standard hosted runners perform static Dockerfile validation
  only.
- Add `CONTRIBUTING.md`, `SECURITY.md`, and issue templates.
- Perform a repository privacy and history audit.
- [x] Preserve multiple audio streams, subtitles, chapters, attachments, data
  streams, and metadata. Reject incompatible output containers before
  inference and expose only atomically completed outputs.
- Publish the methodology, sanitized raw results, and final tables.
- Publish a versioned GitHub Release.

## Later

- Improve the media contract: VFR, rotation, SAR/DAR, duration, and missing
  `nb_frames`.
- Add P010/HDR metadata passthrough, tonemap, and color management.
- Consider a VapourSynth backend as a product feature.
- Consider RIFE/frame interpolation after stabilizing the video contract.
