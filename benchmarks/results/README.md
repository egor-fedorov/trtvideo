# Published Benchmark Results

This directory contains compact, privacy-reviewed benchmark snapshots. Each
snapshot retains the environment, immutable asset identifiers, methodology,
per-run throughput values, aggregate resource metrics, and quality results
needed to interpret the published tables.

Large raw outputs, tensor captures, engines, models, and profiler time series
remain outside Git.

## Baselines

- [RTX 3090 multi-resolution parity benchmark](rtx-3090/README.md) -
  validated `720p -> 1440p` and `1080p -> 4K` RealESRGAN_x2plus and SPAN
  campaigns measured on runtime revision
  `0fc30377046d2c40207d143b1239d8f24e46e7d4`, including quality gates,
  TensorRT ceilings, and an Nsight diagnostic.
