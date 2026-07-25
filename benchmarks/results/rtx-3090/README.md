# RTX 3090 Multi-Resolution Parity Benchmark

This snapshot publishes the validated RealESRGAN_x2plus and SPAN parity
campaigns executed on 2026-07-25. All measured processes and benchmark images
used clean measurement revision
`0fc30377046d2c40207d143b1239d8f24e46e7d4`, which includes the media
preservation path.

The Git commit containing this file is the publication revision. Commits
between the measurement and publication changed benchmark orchestration,
documentation, and tests, not the production `ai_media` runtime. The measured
image identity remains `0fc3037`; relabeling it with a later commit would break
reproducibility.

Both `720p -> 1440p` and `1080p -> 4K` campaigns are valid and
publication-ready under
[`benchmarks/methodology.md`](../../methodology.md). The canonical input is
video-only, so the preservation implementation participates in the measured
startup/finalize path without adding ancillary streams to this workload.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6.
- CPU: AMD Ryzen 5 5600, 6 cores / 12 logical CPUs.
- NVIDIA driver: 595.71.05.
- Project runtime: TensorRT 11.0.0.114, CUDA 13.0, Python 3.12.3.
- Stock VSGAN runtime: TensorRT 10.16.
- Input: Sintel, 24 FPS, limited-range BT.709 H.264 without B-frames.
- Measurement: 100 warmup frames, 1000 measured frames, three rotated rounds.
- Stability: two additional complete rounds when any three-run spread exceeds
  5%; a four-of-five consensus may accept one explicitly reported outlier.
- CUDA Graph: disabled.
- Power policy: no manual power reduction was applied. The active and default
  board limit was 350 W; `sw_power_cap` was observed when the card reached that
  default limit.

## End-To-End Results

| Workload | Input | ai-media | vs-mlrt | VSGAN |
|---|---|---:|---:|---:|
| RealESRGAN_x2plus | 1080p | **2.884 FPS** | 2.394 FPS | 2.399 FPS |
| RealESRGAN_x2plus | 720p | **6.277 FPS** | 5.406 FPS | 5.477 FPS |
| SPAN | 1080p | **25.104 FPS** | 9.348 FPS | 9.018 FPS |
| SPAN | 720p | **49.941 FPS** | 19.825 FPS | 20.315 FPS |

The project was faster than `vs-mlrt` / stock VSGAN by:

- RealESRGAN 1080p: 20.43% / 20.20%;
- RealESRGAN 720p: 16.10% / 14.59%;
- SPAN 1080p: 168.54% / 178.38%;
- SPAN 720p: 151.91% / 145.83%.

### RealESRGAN_x2plus 1080p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **2.884** | **346.79** | **1.003** | 98.59 | 345.04 | **119.65** | 4280.1 | 56.575 |
| vs-mlrt | 2.394 | 417.65 | 1.064 | 82.35 | 309.02 | 129.06 | 4209.1 | 60.436 |
| VSGAN-tensorrt-docker | 2.399 | 416.83 | 1.060 | 83.43 | 311.71 | 129.93 | **4207.1** | 60.436 |

### RealESRGAN_x2plus 720p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **6.277** | **159.32** | **1.006** | 97.03 | 341.17 | **54.39** | **2143.4** | 32.562 |
| vs-mlrt | 5.406 | 184.97 | 1.063 | 84.65 | 307.93 | 56.96 | 2234.4 | 34.906 |
| VSGAN-tensorrt-docker | 5.477 | 182.57 | 1.064 | 85.68 | 310.84 | 56.75 | 2232.4 | 34.906 |

### 2xLiveActionV1_SPAN 1080p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **25.104** | **39.83** | **0.489** | 89.62 | 322.57 | **12.84** | 2664.1 | 56.206 |
| vs-mlrt | 9.348 | 106.97 | 1.240 | 37.75 | 178.77 | 19.12 | 2607.1 | 60.461 |
| VSGAN-tensorrt-docker | 9.018 | 110.89 | 1.224 | 36.64 | 175.64 | 19.48 | **2605.1** | 60.461 |

The `vs-mlrt` result used five runs. Its full spread was 6.10%; rounds 1, 3, 4,
and 5 formed an accepted 2.29% consensus, while round 2 at 9.776 FPS remains
published as an outlier. The headline median and every resource median still
use all five runs.

### 2xLiveActionV1_SPAN 720p

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **49.941** | **20.02** | **0.591** | 79.70 | 298.88 | **5.98** | 1511.7 | 32.860 |
| vs-mlrt | 19.825 | 50.44 | 1.245 | 36.84 | 173.41 | 8.75 | 1496.4 | 35.533 |
| VSGAN-tensorrt-docker | 20.315 | 49.22 | 1.255 | 37.23 | 175.39 | 8.63 | **1494.4** | 35.533 |

## TensorRT Ceiling

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

## Quality Gates

All model-space and product-output gates passed. Product-output metrics compare
all 1000 decoded MP4 frames against the `ai-media-enhancer` output.

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

## Nsight Diagnostic

A 120-frame SPAN 1080p trace used an engine rebuilt on the profiled RTX 3090.
CUDA kernel intervals covered 98.17% of the frame-loop interval. NVDEC and
NVENC workloads overlapped CUDA kernels by 91.25% and 98.92%, respectively. The
trace contained no material per-frame host-to-device or device-to-host
transfer. Device-to-device copies averaged 39.03 MiB and 0.106 ms per frame.

Profiler overhead makes trace FPS non-publishable. The trace is evidence for the
GPU-resident architecture, not another throughput result.

## Bitrate Note

All products used the same H.264 P4/HQ single-pass CBR contract, with GOP 24,
zero B-frames, disabled lookahead/AQ, and resolution-specific target bitrates:
35 Mbps for 720p input and 60 Mbps for 1080p input. Actual bitrate differs
because the FFmpeg NVENC path inserts filler NAL units for strict CBR while the
PyNvVideoCodec path does not. This does not invalidate decode or quality gates,
but it remains visible in the end-to-end resource tables.

## Published Data

[`results.json`](results.json) contains the machine-readable environment,
contracts, asset hashes, per-run FPS, stability decisions, resource/lifecycle
metrics, quality results, TensorRT ceilings, and compact Nsight findings.
Multi-gigabyte MP4 files, FP32 tensor captures, NVML time series, engines,
models, and profiler traces remain outside Git.

Best-tuned and live-action confirmation campaigns remain future work and must
be published separately from this parity snapshot.
