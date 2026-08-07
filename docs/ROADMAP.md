# Roadmap

This file lists only incomplete project priorities. Completed implementation
work belongs in `docs/CHANGES.md`, measured changes in
`docs/PERFORMANCE_LOG.md`, the benchmark contract in
`benchmarks/methodology.md`, and published evidence in
`benchmarks/results/`.

## 1. Open-Source Release

- audit dependency, model, and benchmark-media licenses and document any
  redistribution restrictions;
- enable GitHub private vulnerability reporting and verify repository topics
  and issue forms when the repository becomes public;
- move full production and benchmark image builds to a larger or self-hosted
  GitHub Actions runner; hosted runners continue to perform static Dockerfile
  validation;
- publish versioned production images to GHCR from a trusted release-only
  runner after the redistribution-license audit, and record each immutable
  image digest in its GitHub release; benchmark images remain internal;
- publish the first public versioned GitHub release.

## 2. Later

- improve the media contract for VFR, rotation, SAR/DAR, duration, and missing
  `nb_frames`;
- add P010/HDR metadata handling, tonemapping, and color management;
- add runtime dynamic-shape inference if it provides a concrete workflow
  benefit over static resolution-specific engines.
