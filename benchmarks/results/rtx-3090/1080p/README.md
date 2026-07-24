# RTX 3090 1080p Parity Baseline

This snapshot publishes the validated `1080p -> 4K` parity campaign executed on
2026-07-24. It is the primary baseline for revision
`49ae95a6ef34fe6affb4816855eb9e2cec3421ae`.

The campaign is valid and publication-ready under
[`benchmarks/methodology.md`](../../../methodology.md). The planned 720p
confirmation, Nsight Systems trace, live-action confirmation, and best-tuned
campaign remain pending, so this snapshot is not yet the final multi-workload
release claim.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6.
- CPU: AMD Ryzen 5 5600, 6 cores / 12 logical CPUs.
- NVIDIA driver: 595.71.05.
- Project runtime: TensorRT 11.0.0.114, CUDA 13.0, Python 3.12.3.
- Stock VSGAN runtime: TensorRT 10.16.
- Input: Sintel, 1920x1080, 24 FPS.
- Measurement: 100 warmup frames, 1000 measured frames, three rotated rounds.
- CUDA Graph: disabled.
- Power policy: no manual power reduction was applied. The active and default
  board limit was 350 W; the GPU-reported maximum configurable limit was 375 W.
  `sw_power_cap` was observed under load because the card reached its default
  board limit, not because the benchmark imposed a reduced limit.

## End-To-End Results

### RealESRGAN_x2plus

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **2.790** | **358.37** | **1.005** | 98.90 | 346.33 | **124.12** | 4281.7 | 56.575 |
| vs-mlrt | 2.306 | 433.57 | 1.063 | 81.63 | 315.11 | 136.62 | 4209.1 | 60.436 |
| VSGAN-tensorrt-docker | 2.310 | 432.98 | 1.061 | 82.80 | 317.56 | 137.50 | **4207.1** | 60.436 |

`ai-media-enhancer` was 20.98% faster than `vs-mlrt` and 20.82% faster than
stock VSGAN. Diagnostic `trtexec` reached 2.826 QPS, giving 98.74% pipeline
efficiency.

### 2xLiveActionV1_SPAN

| Implementation | Median FPS | Wall, s | CPU cores | GPU util, % | Power, W | J/frame | Peak VRAM, MiB | Bitrate, Mbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-media-enhancer | **24.601** | **40.65** | **0.490** | 90.91 | 328.10 | **13.32** | 2664.1 | 56.206 |
| vs-mlrt | 9.000 | 111.11 | 1.241 | 36.32 | 205.90 | 22.88 | 2607.1 | 60.461 |
| VSGAN-tensorrt-docker | 9.055 | 110.43 | 1.233 | 36.83 | 205.52 | 22.71 | **2605.1** | 60.461 |

`ai-media-enhancer` was 173.35% faster than `vs-mlrt` and 171.67% faster than
stock VSGAN. Diagnostic `trtexec` reached 27.351 QPS, giving 89.94% pipeline
efficiency. The project used about 0.49 average CPU cores versus approximately
1.24 for both VapourSynth paths.

## Quality Gates

All model-space and product-output gates passed for frames 0, 499, and 999.
Product-output metrics compare all 1000 decoded MP4 frames against the
`ai-media-enhancer` output:

| Workload | Candidate | PSNR, dB | SSIM | Status |
|---|---|---:|---:|---|
| RealESRGAN_x2plus | vs-mlrt | 48.957 | 0.995281 | valid |
| RealESRGAN_x2plus | VSGAN | 48.957 | 0.995281 | valid |
| SPAN | vs-mlrt | 50.886 | 0.996312 | valid |
| SPAN | VSGAN | 50.886 | 0.996312 | valid |

## Bitrate Note

All products requested the same 60 Mbps CBR, 120 Mbit VBV, P4/HQ, GOP 24, and
zero B-frames contract. Actual bitrate differs because the FFmpeg NVENC path
inserts filler NAL units for strict CBR while the PyNvVideoCodec path does not.

For the retained SPAN output, the raw H.264 sizes were:

- `ai-media-enhancer`: 292,738,602 bytes;
- VapourSynth/FFmpeg: 314,901,390 bytes;
- VapourSynth/FFmpeg after removing filler NAL units: 298,130,299 bytes.

Filler accounts for about 76% of the raw bitstream size difference. This does
not invalidate decode or quality gates, but it is an implementation-level
encoder difference and must remain visible when comparing end-to-end results.
The production encoder is not padded solely to make benchmark file sizes equal.

## Published Data

[`results.json`](results.json) contains the machine-readable environment,
contracts, asset hashes, per-run FPS, median resource metrics, quality results,
and diagnostic TensorRT ceilings. Multi-gigabyte MP4, FP32 tensor captures,
NVML time series, and engine/model files remain outside Git.
