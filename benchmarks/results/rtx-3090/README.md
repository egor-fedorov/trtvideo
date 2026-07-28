# RTX 3090 Comparative Benchmark

This directory retains the current RTX 3090 benchmark evidence.
Upstream-default scheduling is the primary publishable product comparison. The
previous tuned snapshot is preserved for audit but withdrawn from publication
pending a corrected VSGAN sweep.

Result classes remain separate:

- `tuned` and `upstream-default` were measured at revision `7aa3d6e`;
- `trtexec` and Nsight diagnostics were measured at revision `0fc3037`.

The classes use the same RTX 3090, Ryzen 5 5600, models, source clips, output
contract, and acceptance policy. They do not form a before/after series:
diagnostics used driver 595.71.05, while upstream-default and tuned used driver
595.84.

## Withdrawn Tuned Snapshot

This snapshot is not a publishable best-tuned comparison. VSGAN was tested only
with four VapourSynth threads, while vstrt used the runtime default of 12 on the
Ryzen 5 5600. The corrected contract sweeps VSGAN `num_streams=2..6` with
runtime-default VapourSynth threads. The measured values below remain historical
evidence and must not support tuned product claims until that sweep, winner
quality gates, and final rotated campaigns are complete.

The tuned campaigns were measured on 2026-07-27 from clean revision
`7aa3d6eea986d5266dbf7a86379e8e4241375335`. Every accepted manifest records
the same RTX 3090, Ryzen 5 5600, driver 595.84, and active 350 W board limit.
No reduced power cap was applied.

RealESRGAN used 30 warmup and 1000 measured frames. SPAN used 100 warmup and
1000 measured frames. Each final result contains three rotated rounds with ten
seconds idle between measured processes. All selected candidates passed
model-space checks and the complete 1000-frame product-output gate.

### End-To-End Throughput

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.130 FPS | **6.305 FPS** | 6.195 FPS | -2.78% |
| RealESRGAN_x2plus | 1080p | 2.810 FPS | **2.850 FPS** | 2.814 FPS | -1.40% |
| SPAN | 720p | 49.850 FPS | **56.554 FPS** | 49.612 FPS | -11.85% |
| SPAN | 1080p | 24.752 FPS | **25.559 FPS** | 21.342 FPS | -3.16% |

RealESRGAN and SPAN 1080p are in parity with the fastest tuned external
implementation under the predefined +/-5% threshold. SPAN 720p is a confirmed
tuned `vs-mlrt` advantage.

These rows describe the withdrawn bounded grid only. In particular, the SPAN
1080p delta against VSGAN is not a tuned product claim. VSGAN builds its own
engine from the same ONNX because TensorRT 10.16 cannot load the project's
TensorRT 11 engine.

### Resource Medians

| Workload | Input | Implementation | FPS | CPU cores | GPU util | Power | J/frame | Peak VRAM | Bitrate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RealESRGAN | 720p | trtvideo | 6.130 | 1.008 | 97.50% | 335.65 W | 54.78 | 2251.7 MiB | 32.562 Mbps |
| RealESRGAN | 720p | vs-mlrt | **6.305** | 2.125 | 98.21% | 337.51 W | **53.53** | 3764.4 MiB | 34.906 Mbps |
| RealESRGAN | 720p | VSGAN | 6.195 | 3.945 | 97.16% | 336.22 W | 54.26 | 6792.4 MiB | 34.906 Mbps |
| RealESRGAN | 1080p | trtvideo | 2.810 | 1.004 | 98.73% | 339.30 W | 120.77 | **4282.1 MiB** | 56.575 Mbps |
| RealESRGAN | 1080p | vs-mlrt | **2.850** | 2.134 | 99.13% | 340.26 W | **119.39** | 7625.1 MiB | 60.436 Mbps |
| RealESRGAN | 1080p | VSGAN | 2.814 | 4.003 | 98.83% | 339.62 W | 120.68 | 14461.1 MiB | 60.436 Mbps |
| SPAN | 720p | trtvideo | 49.850 | **0.596** | 82.20% | 294.54 W | 5.90 | **1507.7 MiB** | 32.860 Mbps |
| SPAN | 720p | vs-mlrt | **56.554** | 6.472 | 90.71% | 309.49 W | **5.47** | 4762.4 MiB | 35.533 Mbps |
| SPAN | 720p | VSGAN | 49.612 | 4.179 | 81.66% | 297.41 W | 5.99 | 3936.4 MiB | 35.533 Mbps |
| SPAN | 1080p | trtvideo | 24.752 | **0.491** | 90.14% | 314.90 W | 12.72 | **2666.1 MiB** | 56.206 Mbps |
| SPAN | 1080p | vs-mlrt | **25.559** | 6.558 | 93.99% | 320.14 W | **12.52** | 9905.1 MiB | 60.461 Mbps |
| SPAN | 1080p | VSGAN | 21.342 | 4.196 | 82.48% | 294.38 W | 13.79 | 8087.1 MiB | 60.461 Mbps |

Tuned `vs-mlrt` trades CPU and VRAM for concurrency. `trtvideo` uses
substantially less host CPU and device memory.

### SPAN 720p Diagnosis

SPAN 720p is the only tuned row outside parity. `trtvideo` reached 49.850 FPS
at 82.20% GPU utilization. The separate `trtexec` diagnostic measured a 62.846
QPS TensorRT ceiling. It used an earlier revision and driver and is retained as
an inference-only reference, not combined with the product campaign.

Startup accounts for 2.569 seconds of the 20.060-second tuned wall time, or
12.81%. The tuned external runners started in approximately one second. Holding
the measured frame-loop and finalize time fixed while reducing project startup
to one second would produce about 54.1 FPS, within 5% of the measured
56.554 FPS `vs-mlrt` result. This counterfactual does not replace the measured
11.85% loss; it identifies startup as a concrete optimization target rather
than evidence of a steady-state GPU-pipeline defect.

### Quality Gates

All winner profiles passed model-space validation on frames 0, 499, and 999.
Product-output metrics compare every decoded frame of the 1000-frame outputs.

| Workload | Input | Candidate | PSNR | SSIM | Status |
|---|---|---|---:|---:|---|
| RealESRGAN_x2plus | 720p | vs-mlrt | 48.251 dB | 0.994583 | valid |
| RealESRGAN_x2plus | 720p | VSGAN | 48.251 dB | 0.994583 | valid |
| RealESRGAN_x2plus | 1080p | vs-mlrt | 48.957 dB | 0.995281 | valid |
| RealESRGAN_x2plus | 1080p | VSGAN | 48.957 dB | 0.995281 | valid |
| SPAN | 720p | vs-mlrt | 49.337 dB | 0.995783 | valid |
| SPAN | 720p | VSGAN | 49.337 dB | 0.995783 | valid |
| SPAN | 1080p | vs-mlrt | 50.886 dB | 0.996312 | valid |
| SPAN | 1080p | VSGAN | 50.886 dB | 0.996312 | valid |

### Tuning Sweep

Sweep FPS selected candidates and was not used as the final comparative value.
Each selected pair advanced to a separate quality gate and rotated campaign.
However, the VSGAN grid below varied CUDA Graph only and did not vary
VapourSynth threads or TensorRT stream count, which is why this snapshot is now
withdrawn.

#### RealESRGAN_x2plus

| Candidate | Configuration | 720p | 1080p |
|---|---|---:|---:|
| `vstrt-s2-g0` | requests auto, streams 2, VS threads auto, graph off | **6.310** | **2.853** |
| `vstrt-s3-g0` | requests auto, streams 3, VS threads auto, graph off | 6.256 | 2.835 |
| `vstrt-s4-g0` | requests auto, streams 4, VS threads auto, graph off | 6.190 | 2.827 |
| `vsgan-s4-g0` | requests auto, streams 4, VS threads 4, graph off | **6.209** | **2.812** |
| `vsgan-s4-g1` | requests auto, streams 4, VS threads 4, graph on | 6.185 | 2.808 |

The RealESRGAN search peaks at two `vs-mlrt` streams. CUDA Graph does not
improve VSGAN for this model.

#### SPAN

| Candidate | Configuration | 720p | 1080p |
|---|---|---:|---:|
| `vstrt-s2-g0` | requests auto, streams 2, VS threads auto, graph off | 43.492 | 16.962 |
| `vstrt-s3-g0` | requests auto, streams 3, VS threads auto, graph off | 53.062 | 22.150 |
| `vstrt-s4-g0` | requests auto, streams 4, VS threads auto, graph off | 56.415 | 24.833 |
| `vstrt-s5-g0` | requests auto, streams 5, VS threads auto, graph off | **56.693** | **25.685** |
| `vstrt-s6-g0` | requests auto, streams 6, VS threads auto, graph off | 56.245 | 25.670 |
| `vsgan-s4-g0` | requests auto, streams 4, VS threads 4, graph off | 50.293 | 21.812 |
| `vsgan-s4-g1` | requests auto, streams 4, VS threads 4, graph on | **50.507** | **21.920** |

Five `vs-mlrt` streams form an interior peak at both resolutions. The sixth
stream does not improve throughput. CUDA Graph gives VSGAN a small positive
selection result on SPAN.

## Upstream-Default Results

These campaigns share the tuned measurement revision, environment, frame
budgets, rotation, and quality policy. `vs-mlrt` uses automatic vspipe
requests, one TensorRT stream, runtime-default VapourSynth threads, and no CUDA
Graph. VSGAN uses automatic requests, four TensorRT streams, four VapourSynth
threads, and no CUDA Graph. `trtvideo` uses its regular `nvcodec` path with
CUDA Graph disabled.

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.134 FPS | 5.739 FPS | **6.198 FPS** | -1.02% |
| RealESRGAN_x2plus | 1080p | 2.817 FPS | 2.448 FPS | **2.817 FPS** | -0.01% |
| SPAN | 720p | 49.948 FPS | 26.910 FPS | **50.873 FPS** | -1.82% |
| SPAN | 1080p | **24.769 FPS** | 9.786 FPS | 21.986 FPS | +12.66% |

The first three rows are parity under the fixed +/-5% criterion. SPAN 1080p is
a confirmed project advantage over the fastest upstream-default external
configuration.

## Diagnostics

### TensorRT Ceiling

`trtexec` is an inference-only diagnostic, not a product competitor.

| Workload | Input | trtexec median |
|---|---|---:|
| RealESRGAN_x2plus | 1080p | 2.921 QPS |
| RealESRGAN_x2plus | 720p | 6.458 QPS |
| SPAN | 1080p | 28.490 QPS |
| SPAN | 720p | 62.846 QPS |

CUDA Graph and data transfers were disabled for these diagnostic runs.

### Nsight Diagnostic

A 120-frame SPAN 1080p trace used an engine rebuilt on the profiled RTX 3090.
CUDA kernel intervals covered 98.17% of the frame-loop interval. NVDEC and
NVENC workloads overlapped CUDA kernels by 91.25% and 98.92%, respectively. The
trace contained no material per-frame host-to-device or device-to-host
transfer. Device-to-device copies averaged 39.03 MiB and 0.106 ms per frame.

Profiler overhead makes trace FPS non-publishable. The trace is evidence for
the GPU-resident architecture, not another throughput result.

### Bitrate Note

All products used the same H.264 P4/HQ single-pass CBR contract, with GOP 24,
zero B-frames, disabled lookahead/AQ, and resolution-specific target bitrates:
35 Mbps for 720p input and 60 Mbps for 1080p input. Actual bitrate differs
because the FFmpeg NVENC path inserts filler NAL units for strict CBR while the
PyNvVideoCodec path does not. This does not invalidate decode or quality gates,
but it remains visible in the resource tables.

## Published Data

[`index.json`](index.json) defines the result-set composition and records each
file's measurement revision and SHA256.

[`tuned.json`](tuned.json) contains every tuning candidate, selected profiles,
winner quality gates, final campaigns, per-run FPS, resources, assets, and
evidence hashes measured at revision `7aa3d6e`.

[`upstream-default.json`](upstream-default.json) contains the documented
upstream-default campaigns, resources, lifecycle metrics, and quality evidence
measured at revision `7aa3d6e`.

[`diagnostics.json`](diagnostics.json) contains the four `trtexec` ceilings,
and compact Nsight findings measured at revision `0fc3037`.

Multi-gigabyte MP4 files, FP32 tensor captures, NVML time series, engines,
models, event logs, and profiler traces remain outside Git. Live-action
confirmation remains future work.
