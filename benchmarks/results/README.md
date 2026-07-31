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

- [RTX 3090 comparative benchmark](rtx-3090/README.md) - validated
  upstream-default, best-tuned, `trtexec`, and Nsight evidence for RealESRGAN
  and SPAN at `720p -> 1440p` and `1080p -> 4K`.

The snapshots were measured after the repository privacy rewrite and corrected
limited-range color path. Every result class records its own clean revision,
hardware, driver, and active power policy; classes from different sessions are
not aggregated. Both current tuned workload matrices are machine-validated and
publishable.
