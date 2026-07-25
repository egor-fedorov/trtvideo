# Published Benchmark Results

This directory contains compact, privacy-reviewed benchmark snapshots. Each
snapshot retains the environment, immutable asset identifiers, methodology,
per-run throughput values, aggregate resource metrics, and quality results
needed to interpret the published tables.

Large raw outputs, tensor captures, engines, models, and profiler time series
remain outside Git.

## Baselines

- [RTX 3090, 1080p to 4K parity baseline](rtx-3090/1080p/README.md) -
  RealESRGAN_x2plus and SPAN on revision
  `49ae95a6ef34fe6affb4816855eb9e2cec3421ae`. This remains valid for that
  revision; a current-release rebaseline is required because the later media
  preservation work changed the measured finalize/mux path.
