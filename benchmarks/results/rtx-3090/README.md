# RTX 3090 Comparative Benchmark

This directory contains the current privacy-reviewed RTX 3090 tuned and
diagnostic evidence. Both result classes were measured on 2026-08-16 from clean
revision `22e31eb660b90fe4bc70d2227987ecb6bf82f38c`, on one RTX 3090 and Ryzen 5
5600 server with driver 595.84 and an active 350 W board limit.

Every workload uses the same pinned CC0 Madrid live-action clip, model, output
contract, and benchmark environment across implementations. Final campaigns
measure 1000 frames in three rotated rounds with ten seconds idle between
processes; RealESRGAN uses 30 warmup frames and SPAN uses 100. Both
cross-resolution matrices and every retained quality report are `valid` and
`publishable`. No retained campaign run recorded a thermal throttle reason;
observed peak temperatures were 51-66 C, and `sw_power_cap` is an allowed
property of the declared 350 W policy. Two rejected thermal attempts remain in
the private raw snapshot and are excluded by the predeclared validity gate.

The raw `software.cuda` field is `null` because this measured revision queried
CPU-only PyTorch for CUDA metadata. The immutable TensorRT base-image digest,
image ID, driver, and TensorRT version remain recorded; this is a metadata
omission, not a change to the measured runtime. Later revisions query the CUDA
Runtime API directly without depending on PyTorch.

## Best-Tuned Results

The adaptive search and full confirmation select external profiles first.
Selected profiles then pass fresh quality gates and independent rotated
campaigns, so reconnaissance FPS is never used as a final comparison value.

| Workload | Input | trtvideo | vs-mlrt | VSGAN | trtvideo vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.140 FPS | 6.271 FPS | **6.345 FPS** | -3.23% |
| RealESRGAN_x2plus | 1080p | 2.803 FPS | **2.830 FPS** | 2.822 FPS | -0.93% |
| SPAN | 720p | **55.760 FPS** | 55.226 FPS | 55.160 FPS | +0.97% |
| SPAN | 1080p | **26.097 FPS** | 25.268 FPS | 25.187 FPS | +3.28% |

All rows fall inside the predeclared +/-5% throughput-parity band.

### Tuned Stream Sweep

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/tuned-sweep-dark.svg">
  <img alt="Tuned TensorRT stream-count sweep for RealESRGAN and SPAN at 720p and 1080p" src="figures/tuned-sweep-light.svg">
</picture>

The lines show one-run, 300-frame reconnaissance measurements. Rings mark the
profiles selected after full 1000-frame confirmation; the dashed `trtvideo`
reference is the independent final-campaign median. Search starts at one stream
and either confirms two declines greater than 1%, reaches the declared
eight-stream boundary, or records a reproducible resource ceiling. Both
RealESRGAN 1080p eight-stream probes reached hashed CUDA out-of-memory ceilings;
the graph records those limits instead of inventing FPS points.

| Workload | Input | vs-mlrt winner | VSGAN winner | Search completion |
|---|---|---|---|---|
| RealESRGAN_x2plus | 720p | streams 2 | streams 2 | decline confirmed |
| RealESRGAN_x2plus | 1080p | streams 2 | streams 2 | resource ceiling |
| SPAN | 720p | streams 4 | streams 4 | range exhausted |
| SPAN | 1080p | streams 6 | streams 5 | range exhausted |

Every selected profile uses automatic vspipe requests, runtime-default
VapourSynth threads, and CUDA Graph disabled. Stage 2 confirms the three
strongest reconnaissance points; within that fully measured shortlist, the
tie-break chooses the lowest stream count within 1% of confirmed peak
throughput, deliberately favoring the external implementation's CPU and VRAM
use.

### Intra-Session Reproducibility

Selected external profiles were measured independently during confirmation and
again in the final rotated campaign. The largest absolute median difference is
0.357%:

| Workload | Input | vs-mlrt final vs confirmation | VSGAN final vs confirmation |
|---|---|---:|---:|
| RealESRGAN_x2plus | 720p | -0.120% | -0.030% |
| RealESRGAN_x2plus | 1080p | -0.160% | +0.256% |
| SPAN | 720p | -0.357% | +0.146% |
| SPAN | 1080p | +0.144% | +0.345% |

This is a same-session harness control, not another product comparison.
`trtvideo` is not a tuning candidate and therefore has no confirmation run.

### Resource Medians

CPU is attributed to the measured child-process tree through
`getrusage(RUSAGE_CHILDREN)`, not to total host activity.

| Workload | Input | Implementation | FPS | CPU cores | GPU util | Power | J/frame | Peak VRAM | Bitrate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RealESRGAN | 720p | trtvideo | 6.140 | **1.010** | 98.70% | 338.27 W | 55.06 | **2233.9 MiB** | 34.484 Mbps |
| RealESRGAN | 720p | vs-mlrt | 6.271 | 2.167 | 98.02% | 337.19 W | 53.77 | 3764.6 MiB | 34.961 Mbps |
| RealESRGAN | 720p | VSGAN | **6.345** | 2.168 | 98.00% | 338.37 W | **53.33** | 3762.6 MiB | 34.961 Mbps |
| RealESRGAN | 1080p | trtvideo | 2.803 | **1.007** | 99.48% | 341.42 W | 122.02 | **4268.3 MiB** | 58.971 Mbps |
| RealESRGAN | 1080p | vs-mlrt | **2.830** | 2.167 | 99.05% | 340.51 W | **120.27** | 7625.3 MiB | 59.870 Mbps |
| RealESRGAN | 1080p | VSGAN | 2.822 | 2.169 | 99.02% | 340.06 W | 120.69 | 7623.3 MiB | 59.866 Mbps |
| SPAN | 720p | trtvideo | **55.760** | **0.561** | 93.59% | 316.46 W | 5.67 | **1399.6 MiB** | 34.486 Mbps |
| SPAN | 720p | vs-mlrt | 55.226 | 5.882 | 89.89% | 308.77 W | **5.59** | 3936.6 MiB | 34.903 Mbps |
| SPAN | 720p | VSGAN | 55.160 | 5.930 | 89.47% | 311.17 W | 5.64 | 3934.6 MiB | 34.903 Mbps |
| SPAN | 1080p | trtvideo | **26.097** | **0.475** | 96.36% | 327.22 W | **12.52** | **2652.3 MiB** | 58.850 Mbps |
| SPAN | 1080p | vs-mlrt | 25.268 | 8.145 | 92.91% | 318.07 W | 12.59 | 11755.3 MiB | 59.712 Mbps |
| SPAN | 1080p | VSGAN | 25.187 | 6.991 | 92.41% | 318.54 W | 12.67 | 9903.3 MiB | 59.712 Mbps |

At parity throughput, the fastest external result uses 2.15-17.15x the
attributed CPU and 1.68-4.43x the peak VRAM.

### Same Throughput, Lower Resource Use

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/throughput-resources-dark.svg">
  <img alt="End-to-end throughput, attributed CPU, and peak VRAM for trtvideo versus the fastest external implementation at equivalent throughput" src="figures/throughput-resources-light.svg">
</picture>

Each row compares `trtvideo` with the fastest external implementation. The
throughput panel normalizes that external result to 100% while retaining the
measured FPS in each bar; CPU and VRAM use linear absolute scales.

## Quality Gates

Shared-input inference checks frames 0, 499, and 999. All output tensors are
exact except the separately built TensorRT 10.16 VSGAN engine on RealESRGAN
1080p; its worst output has p99 absolute error 0.000977, RMSE 0.000276, and
71.17 dB PSNR, well inside the declared numerical thresholds. Production
preprocessing differences are recorded separately as a diagnostic rather than
conflated with inference correctness.

Product-output metrics compare all 1000 decoded frames against `trtvideo`:

| Workload | Input | Candidate | PSNR | SSIM | Status |
|---|---|---|---:|---:|---|
| RealESRGAN_x2plus | 720p | vs-mlrt / VSGAN | 44.719 dB | 0.991528 | valid |
| RealESRGAN_x2plus | 1080p | vs-mlrt | 45.173 dB | 0.992480 | valid |
| RealESRGAN_x2plus | 1080p | VSGAN | 45.172 dB | 0.992476 | valid |
| SPAN | 720p | vs-mlrt / VSGAN | 45.229 dB | 0.991896 | valid |
| SPAN | 1080p | vs-mlrt / VSGAN | 45.791 dB | 0.993051 | valid |

vs-mlrt and VSGAN were captured independently with different run manifests,
capture manifests, container image IDs, and engine SHA-256 values. Their tensor
sets and MP4 outputs are byte-identical in three workloads. RealESRGAN 1080p is
not byte-identical across the TensorRT 11 and 10.16 engines, but both numerical
quality gates pass. Per-workload fingerprints and provenance checks are retained
in [`tuned.json`](tuned.json); no output is claimed identical merely because the
rounded PSNR/SSIM values match.

## Diagnostics

### TensorRT Ceiling

`trtexec` is an inference-only diagnostic, not a product competitor.

| Workload | Input | trtexec median |
|---|---|---:|
| RealESRGAN_x2plus | 720p | 6.227 QPS |
| RealESRGAN_x2plus | 1080p | 2.820 QPS |
| SPAN | 720p | 60.788 QPS |
| SPAN | 1080p | 27.512 QPS |

CUDA Graph and data transfers are disabled for the TensorRT ceiling. The tuned
and diagnostic result sets share one clean revision, server, driver, ONNX/build
contract, and 350 W policy. The diagnostics workflow deliberately rebuilt each
TensorRT engine, so serialized engine hashes differ from the tuned campaign.
For that reason the report does not derive a precise pipeline-efficiency ratio
between the two result classes.

### Nsight Systems

A validated 120-frame SPAN 1080p trace confirms the GPU-resident frame loop:

- merged CUDA kernel intervals cover 98.18% of the frame-loop interval;
- NVDEC and NVENC workloads overlap CUDA kernels by 94.25% and 96.64%;
- zero H2D and zero D2H copies occur during the frame loop;
- 480 D2D copies average 14.83 MiB and 0.043 ms per frame;
- the only H2D copy is a 0.797 MiB initialization transfer before the frame loop.

These findings are recomputed from the retained SQLite export by the diagnostic
publication tool. Profiler overhead makes trace FPS non-publishable; the trace
supports the architecture claim rather than adding a throughput result.

## Encoding Note

All products use the same H.264 P4/HQ single-pass CBR contract, GOP 24, zero
output B-frames, and disabled lookahead/AQ. The FFmpeg NVENC path inserts filler
NAL units for stricter CBR while PyNvVideoCodec does not, producing the small
bitrate difference. Full decode, timestamps, color metadata, and quality gates
remain valid.

## Published Data

[`index.json`](index.json) records result composition, revision, and SHA-256.
[`tuned.json`](tuned.json) and [`diagnostics.json`](diagnostics.json) are the two
self-contained result classes in the current Madrid snapshot. The executable
upstream-default profile remains part of the benchmark methodology, but no
legacy media result is mixed into this publication.

All SVG figures are generated from committed `tuned.json` with
`make -C benchmarks figures`; `make -C benchmarks figures-check` verifies
byte-for-byte reproducibility. MP4 outputs, FP32 tensor captures, NVML time
series, engines, models, event logs, and profiler traces remain outside Git;
the compact JSON retains hashes back to that raw evidence.
