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
  `1080p -> 4K`, measured from the pinned CC0 Madrid live-action contract in one
  clean session.

The snapshots were measured after the repository privacy rewrite and corrected
limited-range color path. Every result class records its clean revision,
hardware, driver, active power policy, and raw-evidence hashes. Every tuned
workload matrix is machine-validated and publishable; diagnostic overlap and
copy findings are regenerated from the retained Nsight SQLite export. Hardware
directories are independent sessions and are not aggregated across hosts.
