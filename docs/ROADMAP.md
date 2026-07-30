# Roadmap

This file lists only incomplete project priorities. Completed implementation
work belongs in `docs/CHANGES.md`, measured changes in
`docs/PERFORMANCE_LOG.md`, the benchmark contract in
`benchmarks/methodology.md`, and published evidence in
`benchmarks/results/`.

## 1. Benchmark Follow-Up

- run the adaptive tuned search, winner quality gates, and final 720p/1080p
  campaigns on the benchmark GPU before replacing the exhaustive-search RTX
  3090 snapshot;
- run one confirmation workload on a short live-action clip with substantial
  motion and fine detail;

## 2. Open-Source Release

- audit dependency, model, and benchmark-media licenses and document any
  redistribution restrictions;
- add `CONTRIBUTING.md`, `SECURITY.md`, and issue templates;
- publish the final methodology, privacy-reviewed result tables, and compact
  machine-readable evidence;
- move full production and benchmark image builds to a larger or self-hosted
  GitHub Actions runner; hosted runners continue to perform static Dockerfile
  validation;
- publish the first public versioned GitHub release.

## 3. Later

- improve the media contract for VFR, rotation, SAR/DAR, duration, and missing
  `nb_frames`;
- add P010/HDR metadata handling, tonemapping, and color management;
- add runtime dynamic-shape inference if it provides a concrete workflow
  benefit over static resolution-specific engines.
