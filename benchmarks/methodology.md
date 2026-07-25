# Benchmark Methodology

## Purpose

The benchmark tests whether the GPU-resident
`NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` pipeline retains an advantage
across the complete path from compressed input to a valid MP4 output.

The following result classes are kept separate:

1. `single-stream vstrt parity` - the project and a locally built
   VapourSynth/vstrt use the same TensorRT 11 engine with one vspipe request and
   one TensorRT stream. This compares integration and video pipelines under a
   fixed, reproducible external scheduling contract.
2. `single-stream VSGAN parity` - the project is compared with a pinned upstream
   VSGAN-tensorrt-docker runtime under the same one-request/one-stream contract.
   Both use the same ONNX, but separate native engines because VSGAN runs
   TensorRT 10.16 while the project runs TensorRT 11.
3. `product-default/tuned` - future campaigns use documented upstream defaults
   or separately selected best-performing settings. These results must not be
   mixed with the single-stream baseline.
4. `trtexec diagnostic` - the inference ceiling without decode, colorspace,
   encode, or mux.

Video2X is excluded from the matrix because the available version does not
support the canonical `RealESRGAN_x2plus` and runs a different anime model. Its
FPS cannot support a same-model performance claim.

The project's primary backend is `nvcodec`. The `ffmpeg` backend remains a
diagnostic baseline and does not replace the GPU-resident result.

## Workflow Separation

Benchmark execution is divided by purpose:

1. `project-only regression` runs only `ai-media-enhancer` through the same
   external timer, validation, and resource accounting used by comparisons. It
   supports before/after engineering decisions but cannot establish a
   competitor advantage.
2. `comparative campaign` rotates project, vstrt, and VSGAN runs by round and
   combines them only after both quality gates pass. This is the sole source of
   publishable competitor claims.
3. `diagnostics` includes `trtexec`, Nsight Systems, and per-stage profiling.
   Diagnostic timings are never mixed into project or competitor FPS tables.

The project implementation runner is shared rather than duplicated. Raw output
is isolated under `artefacts/benchmarks/project/`,
`artefacts/benchmarks/comparative/`, and
`artefacts/benchmarks/diagnostics/`.

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

The published single-stream parity baseline requires:

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

The pinned VSGAN runtime cannot load a TRT11 engine. For VSGAN parity, both
engines are built from the same canonical ONNX on the same GPU, with matching
shape, dtype, batch, and builder intent. Engine hashes and TensorRT runtimes
differ and must be shown explicitly. VSGAN is pinned by immutable image digest
and source revision. Only `.vpy` configuration, model/engine mounts, and the
encoder adapter are allowed; the upstream inference stack is unchanged.

The pinned VSGAN image is an upstream runtime, but the published baseline is not
an upstream-default throughput configuration: its `.vpy` adapter deliberately
uses one vspipe request and one TensorRT stream. Likewise, the vstrt runner does
not use vspipe's automatic multi-request default. The baseline therefore
supports only single-stream parity claims, not claims about stock or maximum
competitor throughput.

## VapourSynth Execution Profiles

The vstrt and VSGAN runners expose the same three scheduling profiles:

| Mode | vspipe requests | TensorRT streams | VapourSynth threads | CUDA Graph |
|---|---:|---:|---:|---:|
| vstrt `parity` | 1 | 1 | runtime default | off |
| VSGAN `parity` | 1 | 1 | 8 | off |
| vstrt `upstream-default` | auto | 1 | runtime default | off |
| VSGAN `upstream-default` | auto | 4 | 4 | off |
| either `tuned` | explicit | explicit | explicit | explicit |

`auto` is not converted to a guessed host-dependent integer. The runner omits
the corresponding `vspipe --requests` or `.vpy` thread argument and lets the
pinned VapourSynth runtime resolve its own default. VSGAN's upstream-default
stream and thread counts come from its pinned `inference_config.py`; vstrt keeps
its documented one-stream default.

Preset modes reject conflicting scheduling overrides. Tuned mode requires
explicit values for requests, TensorRT streams, VapourSynth threads, and CUDA
Graph, including explicit `auto` or `--no-cuda-graph` choices. Every resolved
value is written to the plan and measured-run manifest.

The canonical campaign namespace currently represents only `parity`.
Upstream-default and tuned runners may be planned or smoke-tested independently,
but their publishable campaigns require the separate result namespaces and
aggregation contract defined by the next benchmark infrastructure stage.

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

Parity applies to the requested encoder settings, not to implementation-specific
strict-CBR padding. FFmpeg NVENC may insert filler NAL units while
PyNvVideoCodec may produce a lower actual bitrate for the same target. Reports
must publish actual bitrate and output size, retain the fixed 10% bitrate
tolerance, and disclose confirmed filler behavior. The project output is not
padded or assigned a content-dependent target solely to equalize file sizes.

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

Model-space capture is a separate GPU acceptance job and never runs inside a
timed benchmark process. The canonical frames are zero-based indices `0`, `499`,
and `999`. Each implementation stores raw little-endian FP32 planar RGB tensors
with CHW layout at two boundaries:

```text
input:  normalized RGB immediately before TensorRT
output: RGB immediately after TensorRT, before clipping, YUV conversion, or encode
```

The project reference uses its production `NVDEC -> CV-CUDA -> TensorRT` path.
The vstrt and VSGAN captures use `RGBS` immediately before and after
`core.trt.Model`. VapourSynth's physical `G,B,R` plane serialization is
normalized to logical RGB CHW after capture. Every raw tensor is size-checked
and hashed. The report also requires identical input-video and ONNX hashes;
technical vstrt parity requires the exact same serialized engine as the
project. The pinned VSGAN runtime uses its native TRT10.16 engine built from the
same ONNX.
Each capture records its immutable Docker image ID, clean repository revision,
and source state. Captures from different revisions cannot form a valid report.

Acceptance limits are fixed in each workload manifest:

| Stage | RMSE | p99 absolute error | Minimum PSNR |
|---|---:|---:|---:|
| model input | `0.003922` (`1/255`) | `0.011765` (`3/255`) | `48 dB` |
| model output | `0.007843` (`2/255`) | `0.015686` (`4/255`) | `42 dB` |

All values use normalized RGB where `1.0` is the PSNR data range. These limits
allow small decoder/colorspace and TensorRT-version differences but reject a
materially different preprocessing or model result. They must not be changed
after observing GPU results without invalidating and explaining the campaign.
`max_abs` remains in every tensor report as a diagnostic, but it is not an
acceptance limit: a single finite edge pixel is not representative across
millions of tensor elements and differs between NVDEC/CV-CUDA and
BestSource/zimg chroma reconstruction. Non-finite values always fail the gate.

The v2 gate originally used `max_abs` as a hard limit and input p99 `2/255`.
The first RTX 3090 acceptance run showed high aggregate parity (input/output
PSNR `50–54 dB`, RMSE within limits, and valid final-MP4 parity) while isolated
decoder/colorspace edge pixels exceeded that maximum. The affected campaigns
remain evidence for the rejected v2 methodology and are not reused by this
versioned contract.

Run the gate with `make -C benchmarks model-space-parity`. It writes capture
manifests, raw tensors, logs, and `model-space-parity.json` under the ignored
`artefacts/benchmarks/comparative/quality/` tree. A valid report is required in
addition to the product-output quality report before campaign results can be
published.

Product-output parity uses one separate canonical retained-output run per
implementation. These runs use 100 warmup and 1000 output frames but are not
included in the rotated performance statistics. Each candidate MP4 is compared
with the project MP4 through complete FFmpeg decode passes:

```text
average PSNR >= 35 dB
overall SSIM >= 0.95
exactly 1000 compared frames
```

The same zero-based frames `0`, `499`, and `999` are extracted for manual
inspection. Two normalized crop rectangles are fixed in the workload manifest:
the center quarter and an upper-left quarter. FFmpeg decodes each product once
to generate its complete crop matrix. Run manifests, metric logs/statistics,
MP4 hashes, and every PNG crop are hashed in `product-output-parity.json`.

`make -C benchmarks quality-gates` runs both quality jobs. The campaign
aggregator verifies both reports against the measured workload, input, ONNX, and
engine hashes, as well as the exact Docker image IDs and clean repository
revision. For product-output evidence it reloads the original run manifests
rather than trusting only report metadata. A stable campaign becomes
publishable only when both reports are valid and their referenced evidence is
still present.

## Timing Contract

The primary metric is full-process end-to-end FPS. An external monotonic timer
includes startup, decode, colorspace conversion, inference, encode, flush, and
mux.

For a canonical run:

1. A separate discarded process handles 100 warmup frames.
2. A new process handles exactly 1000 measured frames.
3. Three runs are accepted when full relative spread does not exceed 5%.
4. When any implementation exceeds 5%, two complete rotated rounds are added
   for every implementation to preserve equal sample counts and order balance.
5. After five runs, a result is stable when either the full range passes or the
   narrowest four-of-five subset passes the same 5% threshold. The latter is
   reported as `stable-with-one-outlier`, including the excluded round and
   value.
6. Five runs without an accepted four-run consensus remain unstable.
7. Product order rotates between rounds.
8. The same idle interval is observed between runs.

Raw values, median, min/max, and spread `(max - min) / median` are published.
All headline medians and resource statistics use all measured runs, including
an accepted outlier. The four-of-five subset affects only stability acceptance;
its spread and selected rounds are published separately.
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

The single-stream parity table contains:

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
The trace is not collected inside any measured campaign run. It wraps one
ordinary 120-frame SPAN 1080p `nvcodec` process with CUDA Graph and built-in
stage profiling disabled. Opt-in NVTX ranges label initialization, the frame
loop, decode batches, color conversion, TensorRT, NVENC, and mux. Collection
uses CUDA, NVTX, OS-runtime, and NvVideo tracing plus the selected GPU's video
accelerator trace. CPU IP sampling and scheduler context-switch tracing are
disabled to avoid privileged container execution. Profiler-affected FPS is
never published.

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
reconstructing order from directory names. A campaign is publishable only after
both quality gates pass for the same measured revision and asset contracts.
