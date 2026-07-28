# GPU Benchmark Runbook

This runbook is the operational entrypoint for benchmark work on one physical
NVIDIA GPU. The benchmark contract and validity rules are defined in
[`methodology.md`](methodology.md); this file explains how to execute them.

Run every command from the repository root.

## Requirements

The host must provide:

- Docker with GPU access;
- GNU Make and Git;
- Python `>=3.10,<3.13` for the host coordinator;
- enough disk space for the production, benchmark, vstrt, and pinned VSGAN
  images when competitor workflows are selected.

Use `HOST_PYTHON` when `python3` is not the intended interpreter:

```bash
HOST_PYTHON=/usr/bin/python3.12 \
  benchmarks/bin/run-benchmark.sh comparative
```

An actual workflow requires a clean committed worktree. Docker images,
workflow state, quality evidence, and campaigns are tied to that revision.
`--dry-run` does not require Git state, Docker, or a GPU.

Before a publishable run, record the driver, GPU, power limit, clocks, and
thermal state. Keep the same physical GPU and power policy for every product in
one comparison. The workflow checks measured NVML validity but does not change
the host power limit.

## Goal Interface

Choose the result you need. With no filters, one command processes both
canonical models at both input resolutions:

```bash
benchmarks/bin/run-benchmark.sh project
benchmarks/bin/run-benchmark.sh comparative
benchmarks/bin/run-benchmark.sh tuned
benchmarks/bin/run-benchmark.sh diagnostics
```

The coordinator reads [`workflows/canonical.json`](workflows/canonical.json)
and performs the complete ordered lifecycle:

| Goal | Complete lifecycle |
|---|---|
| `project` | Build project images, prepare/verify assets, build TRT11 engines, smoke, then project-only regression campaigns |
| `comparative` | Build all product images and TRT11/TRT10 engines, smoke every product, run quality gates, then rotated comparative campaigns |
| `tuned` | Build and smoke the full matrix, sweep declared candidates, validate selected winners, run final campaigns, then verify both-resolution evidence |
| `diagnostics` | Build project images and TRT11 engines, smoke, run `trtexec` ceilings, and capture the canonical SPAN 1080p Nsight trace when selected |

The default comparative profile is the published single-request/single-stream
`parity` contract. To measure pinned upstream scheduling defaults:

```bash
benchmarks/bin/run-benchmark.sh comparative --mode upstream-default
```

Tuned candidates and selection rules come from the workload-specific contracts
under [`tuning/`](tuning/). The tuned workflow evaluates all declared
candidates, retains disqualifications, runs the full product-output quality gate
only for selected winners, and verifies that 720p and 1080p evidence agree
before publication. Both canonical contracts require the VSGAN
`num_streams=2..6`, runtime-default VapourSynth-thread grid.

Do not resume tuned artifacts created with an older candidate contract. Start
the complete tuned workflow in an empty tuned artifact namespace after
committing and rebuilding the new revision.

## Selecting A Subset

Use a subset for debugging or a targeted regression:

```bash
benchmarks/bin/run-benchmark.sh project \
  --workload realesrgan \
  --variant 1080p

benchmarks/bin/run-benchmark.sh comparative \
  --workload span \
  --variant 720p \
  --mode upstream-default

benchmarks/bin/run-benchmark.sh diagnostics \
  --workload span \
  --variant 1080p
```

Valid workload keys are `realesrgan` and `span`; valid variants are `720p` and
`1080p`. Omitting either filter selects all values.

A single-resolution tuned run is useful evidence but is not a publication
unit. The final tuned matrix check runs only when both resolutions of a
workload are selected.

To measure the SPAN 720p wrapper regression without running competitors:

```bash
benchmarks/bin/run-benchmark.sh project \
  --workload span \
  --variant 720p
```

After the run, inspect the median internal lifecycle intervals:

```bash
jq '.statistics.median_lifecycle_intervals_sec' \
  artefacts/benchmarks/project/liveaction_span_sintel-720p/suite.json
```

This diagnostic breakdown does not replace the suite's external end-to-end FPS
or make a project-only result comparative.

## Preview

Always review an expensive plan first:

```bash
benchmarks/bin/run-benchmark.sh comparative --dry-run
```

The output lists every image build, asset operation, engine build, smoke,
quality gate, and campaign in execution order. It does not execute commands or
create workflow state.

Smoke runs validate complete decode, media structure, timestamps, color,
keyframes, and the declared encoder settings. They record but do not enforce
average bitrate because 120 frames are not a representative CBR averaging
window. Full campaigns and product-output quality gates retain the fixed 10%
bitrate tolerance.

Use another declarative matrix only when intentionally testing different
artifact paths:

```bash
benchmarks/bin/run-benchmark.sh project \
  --matrix benchmarks/workflows/canonical.json \
  --dry-run
```

## Resume And Recovery

After every successful high-level step, the coordinator atomically records
completion under:

```text
artefacts/benchmarks/workflows/
```

Resume the exact workflow after a transient failure:

```bash
benchmarks/bin/run-benchmark.sh comparative --resume
```

The saved context includes the goal, profile, GPU id, matrix hash, repository
revision, and selected combinations. Resume rejects any mismatch and skips only
steps explicitly recorded as successful. It does not infer completion from
arbitrary files.

Use `--state PATH` to isolate repeated experiments:

```bash
benchmarks/bin/run-benchmark.sh project \
  --workload span \
  --variant 1080p \
  --state artefacts/benchmarks/workflows/span-regression.json
```

Low-level campaign and tuned targets maintain their own append-only evidence.
If they request extra rounds, rerun the same goal with `--resume`. If a target
reports a partial output directory, remove only the directory named in that
error and resume.

Do not resume after changing source, the matrix, profile, or selected artifacts.
Commit the new state, rebuild from a clean revision, and start a separate
workflow. Existing published snapshots remain historical evidence and must not
be rewritten.

## Results

Raw benchmark artifacts are intentionally ignored by Git:

```text
artefacts/benchmarks/project/
artefacts/benchmarks/comparative/
artefacts/benchmarks/diagnostics/
artefacts/benchmarks/workflows/
```

Before publishing a comparative or tuned result, verify:

- every suite and campaign reports `valid` and `publishable`;
- model-space and product-output quality reports are valid;
- retained MP4 files fully decode and the visual crop matrix is acceptable;
- engine, image, revision, workload-contract, and GPU identities match;
- no invalid thermal or competing-process reason was recorded;
- all compared products used the documented codec/rate-control contract;
- both resolutions exist when a tuned cross-resolution claim is made.

`trtexec` is a diagnostic inference ceiling, not a competitor row. Nsight
results describe pipeline scheduling and copies; profiler-affected FPS is not a
benchmark result.

Compact, privacy-reviewed publication snapshots belong in
[`results/`](results/README.md). Do not commit raw videos, tensors, profiler
traces, host identifiers, or model/engine files.

## Low-Level Troubleshooting

The goal coordinator delegates to benchmark Make targets. Use them directly
only to reproduce or debug one failed step:

```bash
make -C benchmarks help
```

Common examples:

```bash
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json

make -C benchmarks run-project \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine

make -C benchmarks profile-nsight \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
```

Direct Make targets do not provide complete-matrix planning or top-level
resume state. Prefer `run-benchmark.sh` for normal benchmark operation.
