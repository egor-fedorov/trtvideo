# Benchmark Methodology

## Purpose

The benchmark tests whether the GPU-resident
`NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` pipeline retains an advantage
across the complete path from compressed input to a valid MP4 output.

The following result classes are kept separate:

1. `upstream-default` - each external product uses the scheduling defaults
   recorded from its pinned upstream version.
2. `tuned` - each external product uses a workload-specific configuration
   selected by the declared candidate sweep and independent quality gate.
3. `trtexec diagnostic` - the inference ceiling without decode, colorspace,
   encode, or mux.

Video2X is excluded from the matrix because the available version does not
support the canonical `RealESRGAN_x2plus` and runs a different anime model. Its
FPS cannot support a same-model performance claim.

The project always uses its production GPU-resident video path.

## Workflow Separation

Benchmark execution is divided by purpose:

1. `project-only regression` runs only `trtvideo` through the same
   external timer, validation, and resource accounting used by comparisons. It
   supports before/after engineering decisions but cannot establish a
   competitor advantage.
2. `comparative campaign` rotates project, vstrt, and VSGAN runs by round and
   combines them only after both quality gates pass. This is the sole source of
   publishable competitor claims.
3. `diagnostics` includes `trtexec`, Nsight Systems, and per-stage profiling.
   Diagnostic timings are never mixed into project or competitor FPS tables.

Project and VapourSynth implementations share the same measurement core while
retaining separate command and lifecycle adapters. Raw output is isolated under
`artefacts/benchmarks/project/`,
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

Every comparative campaign requires:

```text
engine SHA256 identical
input/output dtype identical
static input/output shape identical
batch size = 1
full-frame processing
tiling disabled
execution profile recorded
requests/streams/threads recorded
CUDA Graph state recorded
```

The project and vstrt use the same TensorRT 11 serialized engine. The pinned
VSGAN runtime cannot load that engine, so its TensorRT 10.16 engine is built
from the same canonical ONNX on the same GPU, with matching shape, dtype,
batch, and builder intent. Engine hashes and TensorRT runtimes differ and must
be shown explicitly. VSGAN is pinned by immutable image digest and source
revision. Only `.vpy` configuration, model/engine mounts, and the encoder
adapter are allowed; the upstream inference stack is unchanged.

## VapourSynth Execution Profiles

The vstrt and VSGAN runners expose two scheduling profiles:

| Profile | vspipe requests | TensorRT streams | VapourSynth threads | CUDA Graph |
|---|---:|---:|---:|---:|
| vstrt `upstream-default` | auto | 1 | runtime default | off |
| VSGAN `upstream-default` | auto | 4 | 4 | off |
| either `tuned` | explicit | explicit | explicit | explicit |

`auto` is not converted to a guessed host-dependent integer. The runner omits
the corresponding `vspipe --requests` or `.vpy` thread argument and lets the
pinned VapourSynth runtime resolve its own default. VSGAN's upstream-default
stream and thread counts come from its pinned `inference_config.py`; vstrt keeps
its documented one-stream default.

Preset profiles reject conflicting scheduling overrides. Tuned mode requires
explicit values for requests, TensorRT streams, VapourSynth threads, and CUDA
Graph, including explicit `auto` or `--no-cuda-graph` choices. Every resolved
value is written to the plan and measured-run manifest.

`execution_profile` is the canonical scheduling-profile name across Make,
runner CLIs, manifests, quality evidence, and campaign aggregation.
Diagnostic/reference roles remain explicit in document types, implementation
metadata, and report structure rather than in the scheduling profile.

`upstream-default` is a vendor-default baseline, not a maximum-throughput claim.
In particular, vstrt keeps `num_streams=1` even when the GPU is not saturated.
Tuned candidate grids are workload-specific and selected by the canonical
workflow matrix. For vstrt, RealESRGAN tests `num_streams=2/3/4`, while SPAN
tests `num_streams=2/3/4/5/6`; an initial `2/3/4` SPAN sweep ended at an
increasing upper boundary, and boundary runs found an interior maximum at `5`.
For VSGAN, both workloads test `num_streams=2/3/4/5/6` with automatic vspipe
requests, runtime-default VapourSynth threads, and CUDA Graph disabled. The
VSGAN grid also retains its upstream `streams=4`, `threads=4` configuration
with CUDA Graph disabled and enabled as explicit controls. Contract validation
rejects either workload unless the complete automatic-thread VSGAN stream grid
is present.

Tuned selection is a two-stage protocol. The workload-specific candidate set
and common winner rule are fixed in the contract selected by
`benchmarks/workflows/canonical.json` before the canonical measurement. Every
declared candidate receives a collision-free directory and must provide:

1. a stable canonical performance suite;
2. unchanged workload, input, ONNX, engine, image, encoder, and revision hashes;
3. complete output/media validation from every measured run;
4. model-space parity on frames `0`, `499`, and `999` under that candidate's
   exact requests, stream, thread, and CUDA Graph settings.

Only eligible candidates are ranked. The winner is the highest stable median
end-to-end FPS; candidate ID is the deterministic tie-breaker. Missing evidence
invalidates the entire sweep instead of silently reducing the search space.
The selected pair then runs the independent full 1000-frame product-output
gate. A candidate-specific failure is retained as disqualification evidence and
the next eligible point is promoted. Sweep FPS is never published as a final
product comparison.

The project is not silently tuned during this search. Its profile is fixed in
the same contract to the best already verified production configuration:
`nvcodec` with CUDA Graph disabled. Changing that profile creates a new tuning
contract and requires a new sweep.

Selection, full quality, and the rotated campaign are performed separately for
720p and 1080p. A single-resolution final campaign remains evidence only. The
publication matrix is valid only when both resolutions passed full quality on
the same workload, repository revision, and GPU contract. This avoids using a
1080p quality result to justify a 720p performance claim.

Each profile has separate per-implementation, product-output, and campaign
directories.
A campaign stores its profile and exact vstrt/VSGAN argument strings in an
immutable `campaign.config.json`; resume and aggregation reject a changed
configuration. The aggregator also requires every measured and product-output
manifest to use the selected execution profile and unchanged scheduling
parameters. Results from different profiles therefore cannot form one campaign.

External `vspipe | ffmpeg` encoding is normalized to pinned Ubuntu FFmpeg
`7:6.1.1-3ubuntu5`. The upstream binary requires NVENC API 13.1 and driver 610+,
which are unavailable on the benchmark host. This adapter is recorded in
implementation metadata; changing VSGAN internals constitutes a fork.

CUDA Graph is disabled for upstream-default. The current project implementation
does not expose CUDA Graph; external graph-enabled configurations may be
evaluated only as explicit tuned candidates.

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

The output contract applies to requested encoder settings, not to
implementation-specific strict-CBR padding. FFmpeg NVENC may insert filler NAL
units while
PyNvVideoCodec may produce a lower actual bitrate for the same target. Reports
must publish actual bitrate and output size, retain the fixed 10% bitrate
tolerance, and disclose confirmed filler behavior. The project output is not
padded or assigned a content-dependent target solely to equalize file sizes.
The tolerance is enforced for full campaigns and product-output quality runs.
Reduced 120-frame smoke runs record actual bitrate but do not enforce an
average-bitrate threshold: the short encoder window is not rate-control
evidence and is never publishable.

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
The model-space capture and the production NVCodec pipeline import the same
frame processor; capture adds only a synchronized device-to-host copy after the
shared normalized model input and raw model output boundaries.
The vstrt and VSGAN captures use `RGBS` immediately before and after
`core.trt.Model`. VapourSynth's physical `G,B,R` plane serialization is
normalized to logical RGB CHW after capture. Every raw tensor is size-checked
and hashed. The report also requires identical input-video and ONNX hashes;
the vstrt comparison requires the exact same serialized engine as the project.
The pinned VSGAN runtime uses its native TRT10.16 engine built from the same
ONNX.
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
The first RTX 3090 acceptance run showed high aggregate equivalence
(input/output PSNR `50–54 dB`, RMSE within limits, and valid final-MP4
comparisons) while isolated decoder/colorspace edge pixels exceeded that
maximum. The affected campaigns remain evidence for the rejected v2
methodology and are not reused by this versioned contract.

Run the gate with `make -C benchmarks model-space-parity`. It writes capture
manifests, raw tensors, logs, and `model-space-parity.json` under the ignored
`artefacts/benchmarks/comparative/quality/model-space/<profile>/` tree. Captures
use the same requests, streams, threads, and CUDA Graph settings as the selected
campaign profile. The report and campaign aggregator reject mixed profile
evidence. A valid report is required in addition to the product-output quality
report before campaign results can be published.

Product-output parity uses one separate canonical retained-output run per
implementation. These runs use the workload warmup and the complete 1000-frame
canonical clip, independently of the shorter performance window, and are not
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

Canonical frame budgets are workload-specific:

| Workload contract | Warmup frames | Measured frames |
|---|---:|---:|
| RealESRGAN_x2plus v3 | 30 | 1000 |
| LiveAction SPAN v1 | 100 | 1000 |

RealESRGAN retains a 1000-frame measured window because the project's
PyNvVideoCodec encoder cannot request NVENC filler-data insertion. A trial
400-frame contract stayed below the preselected 10% startup-share limit but
failed the fixed output-bitrate acceptance gate. The shorter 30-frame warmup is
retained because it is a separate discarded process and was sufficient to
stabilize the measured workload. SPAN remains unchanged because startup already
accounts for 12.79% of the project's 720p wall time.

For a canonical run:

1. A separate discarded process handles the workload's warmup frame count.
2. A new process handles the workload's fixed measured frame count.
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

Each workload declares `benchmark.contract_version`. Every run manifest records
the corresponding `benchmark_contract_version`; campaign aggregation rejects
missing, non-canonical, or mixed versions. The final `campaign.json` publishes
the version so results from different frame-budget contracts cannot be merged.

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

Project runs also record diagnostic checkpoints inside the coarse startup and
finalize scopes:

```text
pipeline construction
-> video probe
-> media-preservation preflight
-> TensorRT runtime initialization
-> decoder initialization
-> encoder initialization
-> frame loop
-> NVENC drain
-> mux input close
-> output-container finalization
-> cleanup
-> atomic output commit
-> reporting
-> process exit
```

These checkpoints do not change the external wall-time boundaries or headline
FPS. Raw run manifests publish them under `metrics.lifecycle.detailed`; project
suite and campaign summaries publish median checkpoint intervals as
`median_lifecycle_intervals_sec`. External products retain the common three-scope
contract because their internal lifecycle is not instrumented by this project.

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

This is an aggregate consumption metric, not a profiler attribution. For the
project runner it includes production orchestration, decoder callbacks,
compressed-bitstream writes, CUDA runtime/driver CPU time, and final FFmpeg
muxing. A value near `1.0` does not by itself prove that Python or the benchmark
harness keeps one core busy, nor does it distinguish useful host work from
runtime polling. `perf`, `py-spy`, or Nsight Systems must be used to attribute
that CPU time. Lifecycle scopes divide wall time only; they do not currently
divide CPU time by pipeline stage.

## Metrics

Comparative tables contain:

- median end-to-end FPS and wall time;
- median average CPU cores and share of available CPU capacity;
- average power and joules/frame;
- peak VRAM;
- output size and actual bitrate.

`trtexec` QPS is published separately as an inference-only ceiling and is not
combined with a product campaign from another revision or execution contract.

One representative run includes an Nsight Systems trace for checking H2D/D2H
copies, stream gaps, CPU waits, PCIe traffic, and NVDEC/TensorRT/NVENC overlap.
The trace is not collected inside any measured campaign run. It wraps one
ordinary 120-frame SPAN 1080p `nvcodec` process with built-in stage profiling
disabled. The project runtime does not expose CUDA Graph. Opt-in NVTX ranges
label initialization, the frame loop, decode batches, color conversion,
TensorRT, NVENC, and mux. Collection uses CUDA, NVTX, OS-runtime, and NvVideo
tracing plus the selected GPU's video accelerator trace. CPU IP sampling and
scheduler context-switch tracing are disabled to avoid privileged container
execution. Profiler-affected FPS is never published.

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
reconstructing order from directory names. Its immutable config records the
execution profile and runner arguments used by every round. A campaign is
publishable only after product-output evidence matches the same profile and
both quality gates match the measured revision and asset contracts.
