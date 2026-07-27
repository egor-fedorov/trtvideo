# Published Benchmark Results

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

## Results

- [RTX 3090 comparative benchmark](rtx-3090/README.md) - complete validated
  `720p -> 1440p` and `1080p -> 4K` RealESRGAN_x2plus and SPAN matrix.

The RTX 3090 publication contains four result sets:

- workload-specific tuned sweeps and independent winner campaigns;
- documented upstream-default scheduling;
- single-stream parity as a controlled historical baseline;
- separate `trtexec` and Nsight diagnostics.

Upstream-default and tuned results are complete for both models and both
resolutions. The hardware summary presents tuned results first, includes an
explicit cross-class repeatability control for the unchanged project execution
profile, and retains single-stream parity only as an analytical baseline.
Every result set keeps its own measurement revision, driver, contract, quality
evidence, and machine-readable JSON.
