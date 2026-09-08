# Benchmark Results

This directory contains compact, privacy-reviewed benchmark snapshots. Each
snapshot retains the environment, immutable asset identifiers, methodology,
per-run throughput values, aggregate resource metrics, and quality results
needed to interpret the published tables.

Large raw outputs, tensor captures, engines, models, and profiler time series
remain outside Git.

Each hardware directory contains one human-readable summary, a machine-readable
index, and one self-contained JSON file per methodology result class. Result
sets are not divided by measurement date and are never aggregated across
different contracts or revisions.

## Publication Status

- [RTX 4090 comparative benchmark](rtx-4090/README.md) - validated best-tuned,
  `trtexec`, and Nsight evidence for RealESRGAN and SPAN at `720p -> 1440p` and
  `1080p -> 4K`, measured at the stock 450 W board limit.
- [RTX 3090 comparative benchmark](rtx-3090/README.md) - validated best-tuned,
  `trtexec`, and Nsight evidence for RealESRGAN and SPAN at `720p -> 1440p` and
  `1080p -> 4K`, measured at the stock 350 W board limit.

The current publications use the September 7-8 sessions from clean revision
`b5bf437e17813ea2f6fc1b268a479b612f83dc08`: trtvideo, vs-mlrt, and
TheAnimeScripter on TensorRT 11.2.1.2. Earlier VSGAN publications remain in Git
history and are not relabeled as TAS measurements.

The sessions include archived failed attempts and retries: twelve thermal
failures on RTX 3090, and two CUDA launch failures plus a TAS hang on RTX 4090.
None contributes to the published FPS medians. RTX 4090 RealESRGAN 1080p also
retains a valid vs-mlrt outlier under the declared five-round stability policy.
The hardware reports disclose these limitations; this is not evidence of
uninterrupted, failure-free operation.

The snapshots were measured after the repository privacy rewrite and corrected
limited-range color path. Every result class records its clean revision,
hardware, driver, active power policy, and raw-evidence hashes. Every tuned
workload matrix is machine-validated and publishable; diagnostic overlap and
copy findings are regenerated from the retained Nsight SQLite export. Hardware
directories are independent sessions and are not aggregated across hosts.
