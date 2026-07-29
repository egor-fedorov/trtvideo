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

No benchmark snapshot is currently published. The previous RTX 3090 snapshot
predated the corrected limited-range color path and the repository privacy
rewrite, so it was withdrawn rather than presented as current evidence.

New results will be published only after all images and engines are rebuilt from
one clean rewritten revision and the complete quality, comparative, tuned, and
diagnostic workflows pass.
