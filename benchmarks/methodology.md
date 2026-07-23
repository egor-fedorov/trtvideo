# Benchmark Methodology

## Purpose

The benchmark tests whether the GPU-resident
`NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` pipeline retains an advantage
across the complete path from compressed input to a valid MP4 output.

The following result classes are kept separate:

1. `vstrt parity` - the project and a locally built VapourSynth/vstrt use the
   same TensorRT 11 engine. This compares integration and video pipelines.
2. `VSGAN product` - the project is compared with pinned stock
   VSGAN-tensorrt-docker. Both use the same ONNX, but separate native engines:
   stock VSGAN runs TensorRT 10.16 while the project runs TensorRT 11.
3. `trtexec diagnostic` - the inference ceiling without decode, colorspace,
   encode, or mux.

Video2X is excluded from the matrix because the available version does not
support the canonical `RealESRGAN_x2plus` and runs a different anime model. Its
FPS cannot support a same-model performance claim.

The project's primary backend is `nvcodec`. The `ffmpeg` backend remains a
diagnostic baseline and does not replace the GPU-resident result.

## Workloads

Two x2 models are mandatory:

- `RealESRGAN_x2plus` - a heavy, model-bound workload;
- `2xLiveActionV1_SPAN` - a light workload that exposes pipeline overhead.

Both models are exported through Spandrel to static full-frame ONNX. The
canonical tensor contract is batch 1, RGB NCHW, FP32 input/output bindings, a
mixed-FP16 graph, and no tiling.

The primary input is derived from the lossless Sintel trailer in 1080p24 Y4M.
Two video-only H.264 inputs are prepared:

| Mode | Input | Output | Frames | FPS |
|---|---:|---:|---:|---:|
| primary | 1920x1080 | 3840x2160 | 1000 | 24/1 |
| confirmation | 1280x720 | 2560x1440 | 1000 | 24/1 |

The input uses `yuv420p`, limited BT.709, SAR 1:1, zero B-frames, and GOP 24.
Audio, subtitles, chapters, and user metadata are absent. URLs, hashes,
licenses, and attribution are stored in `workloads/`.

After the canonical campaign, the headline workload is repeated on a short
live-action clip with substantial motion and fine detail. These results are
published separately from Sintel.

## Inference Contract

Technical parity requires:

```text
engine SHA256 identical
input/output dtype identical
static input/output shape identical
batch size = 1
full-frame processing
tiling disabled
requests = 1
TensorRT streams = 1
CUDA Graph disabled
```

Stock VSGAN cannot load a TRT11 engine. For product comparison, both engines are
built from the same canonical ONNX on the same GPU, with matching shape, dtype,
batch, and builder intent. Engine hashes and TensorRT runtimes differ and must
be shown explicitly. VSGAN is pinned by immutable image digest and source
revision. Only `.vpy` configuration, model/engine mounts, and the encoder
adapter are allowed; the stock inference stack is unchanged.

External `vspipe | ffmpeg` encoding is normalized to pinned Ubuntu FFmpeg
`7:6.1.1-3ubuntu5`. The upstream binary requires NVENC API 13.1 and driver 610+,
which are unavailable on the benchmark host. This adapter is recorded in
implementation metadata; changing VSGAN internals constitutes a fork.

CUDA Graph is disabled for the parity baseline. The current project
implementation captures only the TensorRT call and remains experimental. A
graph-enabled configuration is evaluated separately in the best-tuned campaign.

## Output Contract

A direct comparison requires matching:

```text
codec and pixel format
NVENC preset and tuning
rate-control mode
target/min/max bitrate and VBV buffer
GOP and B-frames
FPS and frame count
limited BT.709 color metadata
MP4 container, no audio/subtitles/chapters
```

The canonical target is H.264 `yuv420p`, P4/HQ, zero B-frames, a one-second GOP,
35 Mbps for 1440p, and 60 Mbps for 4K. Rate control is explicitly fixed to
single-pass CBR: target/min/max are equal, the VBV buffer holds two seconds of
bitrate, initial occupancy is one second, and lookahead plus spatial/temporal AQ
are disabled.

An output is valid only after a complete decode and validation of resolution,
codec, pixel format, color tags, FPS, duration, frame count, B-frames, keyframe
interval, actual bitrate, and monotonic PTS/DTS. A valid MP4 may be deleted after
SHA256 is calculated; invalid output is retained for diagnosis.

## Quality Contract

Quality is checked at two points:

1. Model-space parity: compare several RGB/float frames before YUV conversion
   and encoding.
2. Product-output parity: compare decoded final MP4 frames with PSNR/SSIM and
   visual crops.

A pixel diff of only the final MP4 conflates model output, colorspace conversion,
and lossy encoding. VMAF or quality claims require a separate reference
degradation dataset and are not inferred from the throughput workload.

## Timing Contract

The primary metric is full-process end-to-end FPS. An external monotonic timer
includes startup, decode, colorspace conversion, inference, encode, flush, and
mux.

For a canonical run:

1. A separate discarded process handles 100 warmup frames.
2. A new process handles exactly 1000 measured frames.
3. At least three runs are executed; two more are added when relative spread
   exceeds 5%.
4. Product order rotates between rounds.
5. The same idle interval is observed between runs.

Raw values, median, min/max, and spread `(max - min) / median` are published.
Startup/context initialization, steady-state frame loop, and finalize/mux are
also recorded; these scopes do not replace full-process wall time. Cold-start
and warm-cache results are not mixed.

Lifecycle scopes use one boundary contract:

```text
process start -> first completed frame -> last completed frame -> process exit
```

- `startup` includes process launch, imports, model/context and video I/O setup,
  and first-frame latency;
- `steady-state frame loop` measures the interval between completion of the first
  and last frames;
- `finalize + mux` includes encoder drain, flush, container finalization, and
  process exit.

The project records frame boundaries directly from its loop using
`time.perf_counter_ns`. For `vspipe | ffmpeg`, the runner observes native
`vspipe --progress` for the first frame and producer-process completion for the
last boundary; the raw Y4M stream is not proxied through Python. The three scopes
exhaustively sum to the same external wall time. Each raw manifest records its
instrumentation method.

Per-stage profiling and CUDA events are diagnostics and remain disabled in the
measured hot path. A successful smoke run with reduced parameters receives
`status: valid` but `publishable: false`.

## CPU Accounting Contract

CPU use is not derived from total host utilization. The runner takes two
`getrusage(RUSAGE_CHILDREN)` snapshots: after the discarded warmup, immediately
before the measured subprocess, and after all of its child processes have
finished.

Published fields:

- `user_time_sec` and `system_time_sec`;
- `total_time_sec`;
- `average_cores = total_time_sec / measured wall_time_sec`;
- `capacity_percent = average_cores / available_logical_cpus * 100`.

The primary metric is `average_cores`: `1.0` means one fully occupied core and
`2.5` means the equivalent of two and a half cores. `capacity_percent` is
normalized by the container's CPU affinity and is a supporting metric.

Accounting includes `upscale`, `vspipe`, FFmpeg, and other completed processes
in the measured pipeline. It excludes the discarded warmup, benchmark
controller, NVML sampler, and unrelated host processes. Unrelated CPU activity
can still increase wall time, so the canonical campaign runs without parallel
load and with identical CPU affinity.

## Metrics

The product/parity table contains:

- median end-to-end FPS and wall time;
- median average CPU cores and share of available CPU capacity;
- average power and joules/frame;
- peak VRAM;
- output size and actual bitrate.

`trtexec` is published separately. Its diagnostic metric is:

```text
pipeline efficiency = ai-media-enhancer end-to-end FPS / trtexec QPS
```

One representative run includes an Nsight Systems trace for checking H2D/D2H
copies, stream gaps, CPU waits, PCIe traffic, and NVDEC/TensorRT/NVENC overlap.
The trace is not collected inside every measured run.

## Environment Contract

All engines and comparable results are built and measured on one physical GPU.
Driver, power limit, clocks, thermal policy, Docker image digest, display state,
and absence of unrelated GPU load are fixed between series.

The runner records an allowlisted environment:

- GPU model, compute capability, and VRAM;
- CPU model and logical core count;
- driver, CUDA, TensorRT, CV-CUDA, PyNvVideoCodec, FFmpeg, and Python versions;
- immutable image references and source revisions;
- power limit, clocks, temperature, and throttle reasons;
- repository commit;
- SHA256 of input, weights, ONNX, engine, and sidecar;
- sanitized commands and benchmark parameters.

Hostname, username, IP address, GPU UUID/serial, container IDs, absolute host
paths, and a complete environment dump are not recorded.

## Validity And Success

A run is invalid when assets/contracts do not match, output validation fails,
unrelated GPU activity is present, thermal/hardware slowdown occurs, the
environment changes, or the per-frame profiler is enabled. Reaching a
predefined software power limit is recorded but does not by itself invalidate a
run.

Success criteria are fixed before results are obtained:

- more than a 5% median FPS advantage is a confirmed speed advantage;
- a result within +/-5% is parity, followed by comparison of CPU, energy/frame,
  VRAM, and UX;
- losing by more than 5% on both workloads requires profiling and optimization
  before making a claim.

An individual suite always remains acceptance data, even with canonical
frames/runs. A comparative result is formed only by the rotated campaign runner.
The runner stores an append-only event log with actual order, UTC timestamps,
and observed idle intervals; the aggregator validates this log instead of
reconstructing order from directory names. Until quality parity is implemented,
even a valid campaign receives `publishable: false`.
