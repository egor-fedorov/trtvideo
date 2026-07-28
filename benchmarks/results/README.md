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

- [RTX 3090 comparative benchmark](rtx-3090/README.md) - validated
  `720p -> 1440p` and `1080p -> 4K` upstream-default and single-stream results,
  plus a retained tuned snapshot pending a corrected VSGAN sweep.

The RTX 3090 publication contains four result sets:

- workload-specific tuned sweeps and independent winner campaigns;
- documented upstream-default scheduling;
- single-stream parity as a controlled historical baseline;
- separate `trtexec` and Nsight diagnostics.

Upstream-default results are complete for both models and both resolutions.
The first tuned snapshot used only four VapourSynth threads for VSGAN and is
therefore non-publishable as a maximum-throughput comparison. Its measurements
remain available for audit while the corrected `num_streams=2..6`,
runtime-default-thread sweep is pending. Every result set keeps its own
measurement revision, driver, contract, quality evidence, and machine-readable
JSON.
