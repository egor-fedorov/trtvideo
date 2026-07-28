# Roadmap

This file lists only incomplete project priorities. Completed implementation
work belongs in `docs/CHANGES.md`, measured changes in
`docs/PERFORMANCE_LOG.md`, the benchmark contract in
`benchmarks/methodology.md`, and published evidence in
`benchmarks/results/`.

## 1. Current-Revision Benchmark

The existing RTX 3090 upstream-default result remains valid evidence for its
recorded revision. The tuned snapshot is retained for audit but is not a
publishable best-tuned comparison because its VSGAN search used a restricted
VapourSynth thread configuration.

Before publishing current performance claims:

- rebuild all benchmark images and TensorRT engines from one clean revision;
- rerun the complete upstream-default matrix after the torch-free NVCodec
  runtime and startup/finalize changes;
- run the corrected tuned matrix for RealESRGAN and SPAN at 720p and 1080p,
  including the VSGAN `num_streams=2..6` grid with runtime-default VapourSynth
  threads;
- pass profile-scoped model-space and product-output quality gates and rotated
  campaigns; full tuned quality runs apply only to the selected winners;
- refresh the four `trtexec` ceilings using the rebuilt engines;
- repeat the representative SPAN 1080p Nsight diagnostic on the current runtime
  to confirm that the GPU-resident path still has no material per-frame
  host/device transfers;
- replace the withdrawn tuned snapshot only after the complete matrix passes
  the machine-checked publication contract.

After the canonical matrix:

- run one confirmation workload on a short live-action clip with substantial
  motion and fine detail;
- inspect the new SPAN 720p lifecycle intervals. Optimize startup or finalize
  only if the current-revision measurement still identifies them as material;
- attribute project CPU use with `perf` or `py-spy` only if the new campaign
  still shows unexplained sustained CPU consumption.

## 2. Open-Source Release

- audit dependency, model, and benchmark-media licenses and document any
  redistribution restrictions;
- perform a repository privacy and Git-history audit;
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
