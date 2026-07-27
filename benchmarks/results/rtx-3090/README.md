# RTX 3090 Comparative Benchmark

This directory publishes all currently validated RTX 3090 results. The primary
result is the best-tuned comparison. Upstream-default scheduling follows it,
while single-stream parity is retained at the end as a controlled analysis of
pipeline overhead rather than a product-throughput claim.

Result classes remain separate:

- `tuned` and `upstream-default` were measured at revision `7aa3d6e`;
- `single-stream parity` was measured at revision `0fc3037`;
- `trtexec` and Nsight diagnostics were measured at revision `0fc3037`.

The classes use the same RTX 3090, Ryzen 5 5600, models, source clips, output
contract, and acceptance policy. They do not form a before/after series:
single-stream parity and diagnostics used driver 595.71.05, while
upstream-default and tuned used driver 595.84. Numbers are not compared
row-by-row across classes except for the explicitly labeled repeatability
control below.

## Best-Tuned Results

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

The tuned matrix does not support a general fastest-product claim. Relative to
the complete pinned VSGAN product, `trtvideo` is within 1.1% on three rows and
15.98% faster on SPAN 1080p. `vs-mlrt` is a technical TensorRT/VapourSynth
target. VSGAN builds its own engine from the same ONNX because TensorRT 10.16
cannot load the project's TensorRT 11 engine.

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

Tuned `vs-mlrt` trades CPU and VRAM for concurrency. `trtvideo` remains close
to its single-stream resource profile and uses substantially less host CPU and
device memory.

### SPAN 720p Diagnosis

SPAN 720p is the only tuned row outside parity. `trtvideo` reached 49.850 FPS
at 82.20% GPU utilization. The separate single-stream diagnostic measured a
62.846 QPS TensorRT ceiling, so the published end-to-end result is about 79% of
that diagnostic ceiling. This cross-class ratio is directional rather than a
same-run comparison because the diagnostic used the earlier revision and
driver.

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

Sweep FPS selects candidates and is not used as the final comparative value.
Each selected pair advances to a separate quality gate and rotated campaign.

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

## Cross-Class Repeatability Control

The production `trtvideo` execution profile did not change between
single-stream parity, upstream-default, and tuned: all use the regular
GPU-resident `nvcodec` path with CUDA Graph disabled. Its independently
measured median FPS remained within 2.6% across the two measurement revisions
and driver versions.

| Workload | Parity | Upstream default | Tuned | Max range |
|---|---:|---:|---:|---:|
| SPAN 720p | 49.941 | 49.948 | 49.850 | 0.20% |
| SPAN 1080p | 25.104 | 24.769 | 24.752 | 1.40% |
| RealESRGAN 720p | 6.277 | 6.134 | 6.130 | 2.34% |
| RealESRGAN 1080p | 2.884 | 2.817 | 2.810 | 2.57% |

This is a control signal for benchmark repeatability, not permission to merge
the result classes. Their revisions, image identities, scheduling contracts,
and drivers remain distinct in the machine-readable snapshots.

## Diagnostics

### TensorRT Ceiling

`trtexec` is an inference-only diagnostic, not a product competitor.

| Workload | Input | trtexec median | Pipeline efficiency |
|---|---|---:|---:|
| RealESRGAN_x2plus | 1080p | 2.921 QPS | 98.73% |
| RealESRGAN_x2plus | 720p | 6.458 QPS | 97.18% |
| SPAN | 1080p | 28.490 QPS | 88.11% |
| SPAN | 720p | 62.846 QPS | 79.46% |

Pipeline efficiency is single-stream project end-to-end FPS divided by
inference-only `trtexec` QPS. CUDA Graph and data transfers were disabled for
these diagnostic runs.

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

## Single-Stream Parity Analysis

The historical parity snapshot was measured on 2026-07-25 from clean revision
`0fc30377046d2c40207d143b1239d8f24e46e7d4`, with driver 595.71.05 and the
same RTX 3090 and Ryzen 5 5600. It deliberately restricts both VapourSynth
runners to `requests=1`, `num_streams=1`, and disabled CUDA Graph.

The table compares end-to-end pipelines under a reproducible single-stream
contract. It does not establish maximum or upstream-default VSGAN/vstrt
throughput. Its purpose is to expose the cost of the single-request
VapourSynth/BestSource/Y4M path relative to the GPU-resident project pipeline.

| Workload | Input | trtvideo | vs-mlrt | VSGAN |
|---|---|---:|---:|---:|
| RealESRGAN_x2plus | 1080p | **2.884 FPS** | 2.394 FPS | 2.399 FPS |
| RealESRGAN_x2plus | 720p | **6.277 FPS** | 5.406 FPS | 5.477 FPS |
| SPAN | 1080p | **25.104 FPS** | 9.348 FPS | 9.018 FPS |
| SPAN | 720p | **49.941 FPS** | 19.825 FPS | 20.315 FPS |

Under this controlled contract, `trtvideo` was 14.59-20.43% faster on
RealESRGAN and 145.83-178.38% faster on SPAN. The large SPAN deltas are
analytical pipeline-overhead measurements, not claims against tuned or
upstream-default external products.

| Workload | Input | Implementation | FPS | CPU cores | GPU util | Power | J/frame | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| RealESRGAN | 720p | trtvideo | **6.277** | **1.006** | 97.03% | 341.17 W | **54.39** | **2143.4 MiB** |
| RealESRGAN | 720p | vs-mlrt | 5.406 | 1.063 | 84.65% | 307.93 W | 56.96 | 2234.4 MiB |
| RealESRGAN | 720p | VSGAN | 5.477 | 1.064 | 85.68% | 310.84 W | 56.75 | 2232.4 MiB |
| RealESRGAN | 1080p | trtvideo | **2.884** | **1.003** | 98.59% | 345.04 W | **119.65** | 4280.1 MiB |
| RealESRGAN | 1080p | vs-mlrt | 2.394 | 1.064 | 82.35% | 309.02 W | 129.06 | 4209.1 MiB |
| RealESRGAN | 1080p | VSGAN | 2.399 | 1.060 | 83.43% | 311.71 W | 129.93 | **4207.1 MiB** |
| SPAN | 720p | trtvideo | **49.941** | **0.591** | 79.70% | 298.88 W | **5.98** | 1511.7 MiB |
| SPAN | 720p | vs-mlrt | 19.825 | 1.245 | 36.84% | 173.41 W | 8.75 | 1496.4 MiB |
| SPAN | 720p | VSGAN | 20.315 | 1.255 | 37.23% | 175.39 W | 8.63 | **1494.4 MiB** |
| SPAN | 1080p | trtvideo | **25.104** | **0.489** | 89.62% | 322.57 W | **12.84** | 2664.1 MiB |
| SPAN | 1080p | vs-mlrt | 9.348 | 1.240 | 37.75% | 178.77 W | 19.12 | 2607.1 MiB |
| SPAN | 1080p | VSGAN | 9.018 | 1.224 | 36.64% | 175.64 W | 19.48 | **2605.1 MiB** |

All model-space and product-output gates passed. The `vs-mlrt` SPAN 1080p
result used five runs: rounds 1, 3, 4, and 5 formed an accepted 2.29% consensus,
while round 2 remains published as an outlier. The headline median and resource
medians use all five runs.

## Published Data

[`index.json`](index.json) defines the result-set composition and records each
file's measurement revision and SHA256.

[`tuned.json`](tuned.json) contains every tuning candidate, selected profiles,
winner quality gates, final campaigns, per-run FPS, resources, assets, and
evidence hashes measured at revision `7aa3d6e`.

[`upstream-default.json`](upstream-default.json) contains the documented
upstream-default campaigns, resources, lifecycle metrics, and quality evidence
measured at revision `7aa3d6e`.

[`parity.json`](parity.json) contains the machine-readable single-stream
campaigns and quality evidence measured at revision `0fc3037`.

[`diagnostics.json`](diagnostics.json) contains the four `trtexec` ceilings,
pipeline-efficiency values, and compact Nsight findings measured at revision
`0fc3037`.

Multi-gigabyte MP4 files, FP32 tensor captures, NVML time series, engines,
models, event logs, and profiler traces remain outside Git. Live-action
confirmation remains future work.
