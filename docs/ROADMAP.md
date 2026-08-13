# Roadmap

This file lists only incomplete project priorities. Completed implementation
work belongs in `docs/CHANGES.md`, measured changes in
`docs/PERFORMANCE_LOG.md`, the benchmark contract in
`benchmarks/methodology.md`, and published evidence in
`benchmarks/results/`.

## 1. Open-Source Release

- enable GitHub private vulnerability reporting and verify repository topics
  and issue forms when the repository becomes public;
- provision an isolated release runner for the protected GHCR workflow; hosted
  runners continue to perform static Dockerfile validation, while benchmark
  images remain internal;
- complete the maintainer review of the recorded release SBOM and publish the
  first versioned production image with its immutable digest;
- publish the first public versioned GitHub release.

## 2. Later

- improve the media contract for VFR, rotation, SAR/DAR, duration, and missing
  `nb_frames`;
- add P010/HDR metadata handling, tonemapping, and color management;
- add runtime dynamic-shape inference if it provides a concrete workflow
  benefit over static resolution-specific engines.
