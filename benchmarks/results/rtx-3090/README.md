# RTX 3090 Comparative Benchmark

This directory publishes all currently validated RTX 3090 results. Result
classes are separated by execution contract rather than by measurement date:

- `single-stream parity` was measured at revision `0fc3037`;
- `upstream-default` and `tuned` were measured at revision `7aa3d6e`.
- `trtexec` and Nsight diagnostics were measured at revision `0fc3037`.

The values are current for their recorded contracts. They are kept on one page
because they use the same GPU, CPU, models, source clips, output contract, and
acceptance policy. Revisions and image identities remain attached to each
class, so numbers from different contracts are never silently merged.

## Single-Stream Parity Baseline

This snapshot publishes validated RealESRGAN_x2plus and SPAN single-stream
parity campaigns executed on 2026-07-25. All measured processes and benchmark
images used clean measurement revision
`0fc30377046d2c40207d143b1239d8f24e46e7d4`, which includes the media
preservation path.

The Git commit containing this file is the publication revision. The measured
image identity remains `0fc3037`; later publication commits do not change the
revision attached to the measurement.

Both `720p -> 1440p` and `1080p -> 4K` campaigns are valid and
publication-ready under
[`benchmarks/methodology.md`](../../methodology.md). The canonical input is
video-only, so the preservation implementation participates in the measured
startup/finalize path without adding ancillary streams to this workload.

### Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6.
- CPU: AMD Ryzen 5 5600, 6 cores / 12 logical CPUs.
- NVIDIA driver: 595.71.05.
- Project runtime: TensorRT 11.0.0.114, CUDA 13.0, Python 3.12.3.
- Pinned upstream VSGAN runtime: TensorRT 10.16.
- Input: Sintel, 24 FPS, limited-range BT.709 H.264 without B-frames.
- Measurement: 100 warmup frames, 1000 measured frames, three rotated rounds.
- Stability: two additional complete rounds when any three-run spread exceeds
  5%; a four-of-five consensus may accept one explicitly reported outlier.
- CUDA Graph: disabled.
- External VapourSynth scheduling: one vspipe request and one TensorRT stream.
  VSGAN used eight explicitly configured VapourSynth threads; the TRT11 vstrt
  runner used the VapourSynth runtime default.
- Power policy: no manual power reduction was applied. The active and default
  board limit was 350 W; `sw_power_cap` was observed when the card reached that
  default limit.

### Claim Boundary

The VSGAN container is pinned to an upstream image, but these measurements do
not use its upstream-default throughput configuration. Both VapourSynth runners
were deliberately restricted to `requests=1`, `num_streams=1`, and disabled
CUDA Graph. The project used its regular production NVDEC/CV-CUDA/TensorRT/NVENC
path.

The table therefore compares end-to-end pipelines under a reproducible
single-stream contract. It does not establish maximum or upstream-default
VSGAN/vstrt throughput. This distinction is especially important for SPAN,
where the VapourSynth runners reached only about 37% GPU utilization.

### End-To-End Results

| Workload | Input | trtvideo | vs-mlrt | VSGAN |
|---|---|---:|---:|---:|
| RealESRGAN_x2plus | 1080p | **2.884 FPS** | 2.394 FPS | 2.399 FPS |
| RealESRGAN_x2plus | 720p | **6.277 FPS** | 5.406 FPS | 5.477 FPS |
| SPAN | 1080p | **25.104 FPS** | 9.348 FPS | 9.018 FPS |
| SPAN | 720p | **49.941 FPS** | 19.825 FPS | 20.315 FPS |

Under this single-stream contract, the project was faster than `vs-mlrt` /
the pinned VSGAN runtime by:

- RealESRGAN 1080p: 20.43% / 20.20%;
- RealESRGAN 720p: 16.10% / 14.59%;
- SPAN 1080p: 168.54% / 178.38%;
- SPAN 720p: 151.91% / 145.83%.

The SPAN deltas expose the overhead of the measured single-request
VapourSynth/BestSource/Y4M path relative to the GPU-resident project pipeline.
They are not a product claim against tuned or upstream-default VSGAN settings.

#### RealESRGAN_x2plus 1080p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trtvideo | **2.884** | **346.79** | **1.003** | 98.59 | 345.04 | **119.65** | 4280.1 | 56.575 |
| vs-mlrt | 2.394 | 417.65 | 1.064 | 82.35 | 309.02 | 129.06 | 4209.1 | 60.436 |
| VSGAN-tensorrt-docker | 2.399 | 416.83 | 1.060 | 83.43 | 311.71 | 129.93 | **4207.1** | 60.436 |

#### RealESRGAN_x2plus 720p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trtvideo | **6.277** | **159.32** | **1.006** | 97.03 | 341.17 | **54.39** | **2143.4** | 32.562 |
| vs-mlrt | 5.406 | 184.97 | 1.063 | 84.65 | 307.93 | 56.96 | 2234.4 | 34.906 |
| VSGAN-tensorrt-docker | 5.477 | 182.57 | 1.064 | 85.68 | 310.84 | 56.75 | 2232.4 | 34.906 |

#### 2xLiveActionV1_SPAN 1080p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trtvideo | **25.104** | **39.83** | **0.489** | 89.62 | 322.57 | **12.84** | 2664.1 | 56.206 |
| vs-mlrt | 9.348 | 106.97 | 1.240 | 37.75 | 178.77 | 19.12 | 2607.1 | 60.461 |
| VSGAN-tensorrt-docker | 9.018 | 110.89 | 1.224 | 36.64 | 175.64 | 19.48 | **2605.1** | 60.461 |

The `vs-mlrt` result used five runs. Its full spread was 6.10%; rounds 1, 3, 4,
and 5 formed an accepted 2.29% consensus, while round 2 at 9.776 FPS remains
published as an outlier. The headline median and every resource median still
use all five runs.

#### 2xLiveActionV1_SPAN 720p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trtvideo | **49.941** | **20.02** | **0.591** | 79.70 | 298.88 | **5.98** | 1511.7 | 32.860 |
| vs-mlrt | 19.825 | 50.44 | 1.245 | 36.84 | 173.41 | 8.75 | 1496.4 | 35.533 |
| VSGAN-tensorrt-docker | 20.315 | 49.22 | 1.255 | 37.23 | 175.39 | 8.63 | **1494.4** | 35.533 |

### TensorRT Ceiling

`trtexec` is diagnostic and is not a product competitor.

| Workload | Input | trtexec median | Pipeline efficiency |
|---|---|---:|---:|
| RealESRGAN_x2plus | 1080p | 2.921 QPS | 98.73% |
| RealESRGAN_x2plus | 720p | 6.458 QPS | 97.18% |
| SPAN | 1080p | 28.490 QPS | 88.11% |
| SPAN | 720p | 62.846 QPS | 79.46% |

Pipeline efficiency is project end-to-end FPS divided by inference-only
`trtexec` QPS. CUDA Graph and data transfers were disabled for these diagnostic
runs.

### Quality Gates

All model-space and product-output gates passed. Product-output metrics compare
all 1000 decoded MP4 frames against the `trtvideo` output.

| Workload | Input | Candidate | PSNR, dB | SSIM | Status |
|---|---|---|---:|---:|---|
| RealESRGAN_x2plus | 1080p | vs-mlrt | 48.957 | 0.995281 | valid |
| RealESRGAN_x2plus | 1080p | VSGAN | 48.957 | 0.995281 | valid |
| RealESRGAN_x2plus | 720p | vs-mlrt | 48.251 | 0.994583 | valid |
| RealESRGAN_x2plus | 720p | VSGAN | 48.251 | 0.994583 | valid |
| SPAN | 1080p | vs-mlrt | 50.886 | 0.996312 | valid |
| SPAN | 1080p | VSGAN | 50.886 | 0.996312 | valid |
| SPAN | 720p | vs-mlrt | 49.337 | 0.995783 | valid |
| SPAN | 720p | VSGAN | 49.337 | 0.995783 | valid |

### Nsight Diagnostic

A 120-frame SPAN 1080p trace used an engine rebuilt on the profiled RTX 3090.
CUDA kernel intervals covered 98.17% of the frame-loop interval. NVDEC and
NVENC workloads overlapped CUDA kernels by 91.25% and 98.92%, respectively. The
trace contained no material per-frame host-to-device or device-to-host
transfer. Device-to-device copies averaged 39.03 MiB and 0.106 ms per frame.

Profiler overhead makes trace FPS non-publishable. The trace is evidence for the
GPU-resident architecture, not another throughput result.

### Bitrate Note

All products used the same H.264 P4/HQ single-pass CBR contract, with GOP 24,
zero B-frames, disabled lookahead/AQ, and resolution-specific target bitrates:
35 Mbps for 720p input and 60 Mbps for 1080p input. Actual bitrate differs
because the FFmpeg NVENC path inserts filler NAL units for strict CBR while the
PyNvVideoCodec path does not. This does not invalidate decode or quality gates,
but it remains visible in the end-to-end resource tables.

## Upstream-Default And Tuned

These campaigns were measured on 2026-07-27 from clean revision
`7aa3d6eea986d5266dbf7a86379e8e4241375335`. Every accepted manifest records
the same RTX 3090, Ryzen 5 5600, driver 595.84, and active 350 W board limit.
No reduced power cap was applied.

RealESRGAN used 30 warmup and 1000 measured frames. SPAN used 100 warmup and
1000 measured frames. Each final result contains three rotated rounds with ten
seconds idle between measured processes. All selected candidates passed
model-space checks and the complete 1000-frame product-output gate.

### Upstream Defaults

`vs-mlrt` uses automatic vspipe requests, one TensorRT stream,
runtime-default VapourSynth threads, and no CUDA Graph. VSGAN uses automatic
requests, four TensorRT streams, four VapourSynth threads, and no CUDA Graph.
The project uses its regular `nvcodec` path with CUDA Graph disabled.

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.134 FPS | 5.739 FPS | **6.198 FPS** | -1.02% |
| RealESRGAN_x2plus | 1080p | 2.817 FPS | 2.448 FPS | **2.817 FPS** | -0.01% |
| SPAN | 720p | 49.948 FPS | 26.910 FPS | **50.873 FPS** | -1.82% |
| SPAN | 1080p | **24.769 FPS** | 9.786 FPS | 21.986 FPS | +12.66% |

The first three rows are parity under the fixed +/-5% criterion. SPAN 1080p is
a confirmed project advantage over the fastest upstream-default external
configuration.

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

### Tuned End-To-End Results

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.130 FPS | **6.305 FPS** | 6.195 FPS | -2.78% |
| RealESRGAN_x2plus | 1080p | 2.810 FPS | **2.850 FPS** | 2.814 FPS | -1.40% |
| SPAN | 720p | 49.850 FPS | **56.554 FPS** | 49.612 FPS | -11.85% |
| SPAN | 1080p | 24.752 FPS | **25.559 FPS** | 21.342 FPS | -3.16% |

RealESRGAN and SPAN 1080p are parity with the fastest tuned external
implementation. SPAN 720p is a confirmed tuned `vs-mlrt` advantage.

#### Tuned Resource Medians

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

Tuned `vs-mlrt` trades CPU and VRAM for concurrency. The project remains close
to its single-stream resource profile and uses substantially less host CPU and
device memory.

### Tuned Quality Gates

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

All winner profiles also passed model-space validation on frames 0, 499, and
999. Product-output metrics compare every decoded frame of the 1000-frame
outputs.

### Tuned Result Boundary

The tuned matrix does not support a general fastest-product claim:

- the project is in parity on RealESRGAN and SPAN 1080p;
- tuned `vs-mlrt` has a confirmed 11.85% advantage on SPAN 720p;
- relative to tuned VSGAN, the project is within 1.1% on three rows and 15.98%
  faster on SPAN 1080p.

`vs-mlrt` is a technical TensorRT/VapourSynth target. The pinned VSGAN image is
the complete external product comparison. VSGAN builds its own engine from the
same ONNX because TensorRT 10.16 cannot load the project's TensorRT 11 engine.

## Published Data

[`index.json`](index.json) defines the result-set composition and records each
file's measurement revision and SHA256.

[`parity.json`](parity.json) contains the machine-readable single-stream
campaigns and quality evidence measured at revision `0fc3037`.

[`upstream-default.json`](upstream-default.json) contains the documented
upstream-default campaigns, resources, lifecycle metrics, and quality evidence
measured at revision `7aa3d6e`.

[`tuned.json`](tuned.json) contains every tuning candidate, selected profiles,
winner quality gates, final campaigns, per-run FPS, resources, assets, and
evidence hashes measured at revision `7aa3d6e`.

[`diagnostics.json`](diagnostics.json) contains the four `trtexec` ceilings,
pipeline-efficiency values, and compact Nsight findings measured at revision
`0fc3037`.

Multi-gigabyte MP4 files, FP32 tensor captures, NVML time series, engines,
models, event logs, and profiler traces remain outside Git. Live-action
confirmation remains future work.
