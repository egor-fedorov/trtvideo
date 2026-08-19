# Performance Log

Historical record of comparable performance changes and cross-hardware scaling
observations. See `README.md` for current commands and `benchmarks/results/` for
publication evidence. New entries must identify the change or observation, the
measurement contract, the gain or regression, and the evidence location. Keep
entries in reverse chronological order.

Only detailed entries that still explain the current runtime are retained here.
Experiments against removed interfaces or superseded media contracts are
summarized at the end; their full text remains available in Git history.

Commit identifiers dated before 2026-07-30 were recorded before the July 2026
privacy rewrite. They remain immutable labels from the original measurement
records but are not expected to resolve in the rewritten history. Current
publication evidence must come from one clean post-rewrite revision.

## 2026-08-18 - Cross-GPU SPAN scaling

Independent, fully valid Madrid sessions measured the current tuned comparison
on an RTX 3090 with a six-core Ryzen 5 5600 and on an RTX 4090 with an eight-core
Ryzen 7 5700X3D. Before the RTX 4090 measurement, the working hypothesis was
that additional host CPU capacity would reduce the external SPAN path's gap.
The observed direction was the opposite.

| SPAN input | RTX 3090 difference | RTX 4090 difference | `trtvideo` scaling | Fastest external scaling | TensorRT ceiling scaling |
|---|---:|---:|---:|---:|---:|
| 720p | +0.97% | +17.89% | 1.79x | 1.53x | 1.92x |
| 1080p | +3.28% | +25.49% | 1.86x | 1.53x | 1.94x |

The fastest external result increased its attributed CPU use from 5.88/8.15
cores at 720p/1080p to 11.83/12.17 cores. Its GPU utilization fell from
89.89-92.91% to 75.62-80.95% while the TensorRT-only ceiling nearly doubled.
Available CPU core count therefore does not explain the RTX 3090 near-parity
result by itself.

The observation is consistent with resolution-dependent, per-frame host
processing and transport overhead becoming a larger fraction of wall time as
inference gets shorter. It does not isolate PCIe transfer latency: CPU, GPU,
power policy, and complete host differ between sessions, and the external path
was not captured in Nsight Systems. The resulting forward hypothesis is limited
to SPAN-like light models where GPU compute improves faster than the
host/transport path; RealESRGAN remains a compute-bound parity result.

Evidence:

- [`benchmarks/results/rtx-3090/`](../benchmarks/results/rtx-3090/README.md)
- [`benchmarks/results/rtx-4090/`](../benchmarks/results/rtx-4090/README.md)

## 2026-07-29 - Corrected color path and streaming mux baseline

The current GPU-resident path explicitly expands limited-range NV12 Y/UV code
values before CV-CUDA RGB conversion and compresses generated NV12 before
NVENC. NVENC packets stream directly into a long-lived FFmpeg mux process;
source-stream preservation, MP4 `faststart`, and atomic output commit remain
enabled.

Benchmark:

- Revision: `ba8d2b0`.
- Workload: canonical SPAN `720p -> 1440p`.
- Pipeline: NVDEC -> range expansion -> CV-CUDA -> TensorRT -> CV-CUDA ->
  range compression -> NVENC -> streaming FFmpeg mux.
- GPU: RTX 3090.
- Workload size: 100 warmup frames and 1000 measured frames.
- Runs: three standalone runs; all passed the canonical output contract.
- Quality: upstream-default model-space and product-output gates passed with
  the shared production processor.

| Metric | Current baseline |
|---|---:|
| Median end-to-end throughput | 56.236 FPS |
| Relative spread | 0.170% |
| Median startup | 0.846 s |
| Median steady-state frame loop | 16.553 s |
| Median finalize | 0.398 s |
| Mux input close + mux finalization | 0.192 s |

This supersedes the earlier standalone streaming-mux numbers, whose outputs
were produced before the limited-range correction. Those measurements are not
retained as an A/B performance claim. A future isolated mux comparison must run
both revisions through the validated color contract.

## 2026-07-29 - PyTorch to direct CUDA runtime

The production frame path moved from PyTorch tensors to direct CUDA Python,
CV-CUDA, and TensorRT bindings. A controlled A/B benchmark compared adjacent
clean revisions:

- PyTorch: `bb226ec23f0a3035e2a583b6664f3d131e262fb1`.
- Direct CUDA: `6fd22b536165a650832b6c0af8657fc4fbf0801f`.
- Workload: canonical SPAN `720p -> 1440p`.
- Pipeline: NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC.
- GPU: RTX 3090 at a 350 W board limit.
- Workload size: 100 warmup frames and 1000 measured frames.
- Runs: three alternating runs per revision; all output validation passed.
- Engine SHA256:
  `d147623c0c5cbf41f303444a71e9c0cadbd2560511d48a16a83b8ce7dc724f29`.
- Input SHA256:
  `7bbef1c5d80ce5452f2b5b61d04a1b94458d044176ab3a2da19984d4a8180062`.

| Metric | PyTorch | Direct CUDA | Change |
|---|---:|---:|---:|
| Median end-to-end throughput | 49.717 FPS | 55.283 FPS | +11.2% |
| Median startup | 2.593 s | 0.816 s | -68.5% |
| Median steady-state frame loop | 16.617 s | 16.483 s | -0.8% |
| Median finalize | 0.900 s | 0.757 s | -15.9% |
| Median CPU use | 0.597 cores | 0.539 cores | -9.8% |
| Median GPU utilization | 82.49% | 90.76% | +8.27 pp |
| Median power | 295.98 W | 313.31 W | +5.9% |
| Median energy per frame | 5.954 J | 5.669 J | -4.8% |

Most of the end-to-end gain comes from removing PyTorch process startup.
Steady-state frame-loop time also improved by 0.8%, so the migration introduced
no measured throughput regression. Higher GPU utilization increased average
power while reducing energy per processed frame.

A later standalone run on the direct-CUDA code measured 55.412 FPS with 0.212%
relative spread, within 0.3% of the controlled result. Its lifecycle data
measured 5 ms for NVENC drain, 113 ms for bitstream close, and 400 ms for the
then-current preserved-media mux. The controlled adjacent-revision comparison
remains the performance evidence.

## Retired Experiments

These measurements informed development but no longer describe the public
runtime or current benchmark contract. Keeping their conclusions here avoids
repeating dead-end experiments without presenting obsolete commands as usable
instructions.

| Date | Experiment | Recorded conclusion | Why detail was retired |
|---|---|---|---|
| 2026-07-27 | Pre-rewrite RTX 3090 comparison | Mixed parity and one SPAN 720p loss | Snapshot was withdrawn and preceded the corrected color path; replaced by current Madrid publications |
| 2026-07-12 | NVENC stream synchronization | Removed host busy-wait with no stable throughput change | Measured the PyTorch runtime with non-current CPU accounting |
| 2026-07-12 | TensorRT 26.04 -> 26.06 | No stable regression; CUDA Graph had no consistent gain | Old Quadro engines and superseded TensorRT/runtime contract |
| 2026-05-12 | TensorRT 26.03 -> 26.04 | No improvement; 1080p regressed in that stack | Old Quadro engines and removed FP16-I/O interface |
| 2026-05-11 | NVDEC/NVENC buffer pool | Preallocation improved processing FPS by 2.6% | Design survives, but the profiler and runtime used for the number do not |
| 2026-05-11 | FP16 I/O | Improved throughput by 4.1% and reduced measured memory by 41.7% | Public model contract now uses FP32 boundaries with mixed FP16 inside ONNX |
| 2026-05-11 | CUDA Graph | No end-to-end gain beyond noise | Product option and legacy runtime were removed |
| 2026-05-11 | Docker dependency cache | Warm code-only rebuild reused dependency layers | No comparable timing was recorded; this is build behavior, not performance evidence |
