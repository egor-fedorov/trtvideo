# Benchmarks

This directory contains reproducible workload manifests, pinned implementation
metadata, isolated Docker environments, and runners. Models, ONNX files,
TensorRT engines, source videos, and raw results are not added to Git.
Compact, privacy-reviewed publication snapshots are stored in `results/`.

- `methodology.md` - comparison classes and validity criteria.
- `workloads/` - RealESRGAN and SPAN workload manifests.
- `workflows/` - canonical workload/resolution matrix for complete workflows.
- `implementations.json` - pinned implementations and execution profiles.
- `docker/` - TensorRT 11 vstrt and pinned VSGAN environments.
- `bin/run-benchmark.sh` - goal-based host workflow entrypoint.
- `scripts/contracts/` - benchmark plans, engine metadata, and run/quality
  evidence contracts shared across runners, gates, aggregation, and tuning.
- `scripts/runtime/` - process timing, CPU/NVML sampling, environment capture,
  command execution, JSON output, media validation orchestration, and suite
  policy.
- `scripts/runners/` - product, VapourSynth, VSGAN, and trtexec adapters built
  on the shared runtime.
- `scripts/diagnostics/` - one-off profiler orchestration outside FPS campaigns.
- `scripts/campaign/` - rotated campaign scheduling and aggregation.
- `scripts/quality/` - shared-input inference, preprocessing diagnostics, and
  final-output quality evidence.
- `scripts/report/` - privacy-reviewed publication export and deterministic SVG
  generation from committed JSON.
- `scripts/tuning/` - adaptive search, deterministic selection, and
  cross-resolution publication checks.
- `scripts/workflow/` - complete goal planning, execution, and resume state.
- `scripts/workloads/` - asset preparation, validation, and engine builders.
- `tuning/candidates.json` - RealESRGAN adaptive search and selection policy.
- `tuning/span_candidates.json` - SPAN adaptive search and selection policy.
- `GPU_RUNBOOK.md` - acceptance sequence on the benchmark GPU.
- [`results/`](results/README.md) - committed benchmark tables and
  machine-readable sanitized snapshots; large raw artifacts remain under
  ignored `artefacts/`.

The benchmark workflow is separated from the root `Makefile`:

```bash
make -C benchmarks help
```

Published figures are generated from the privacy-reviewed JSON snapshot rather
than maintained by hand:

```bash
make -C benchmarks figures
make -C benchmarks figures-check
```

A completed copied session is converted into compact tuned and diagnostic
publication snapshots by benchmark-specific exporters. Source paths are
relative to the repository root:

```bash
make -C benchmarks publish-tuned \
  TUNED_PUBLICATION_SOURCE=artefacts/benchmarks/session/comparative/tuning

make -C benchmarks publish-diagnostics \
  DIAGNOSTICS_PUBLICATION_SOURCE=artefacts/benchmarks/session/diagnostics
```

The tuned exporter requires both valid Madrid cross-resolution matrices, one
clean revision, independent candidate provenance, and passing numerical quality
gates. It records tensor and MP4 identity per workload without requiring
separately built TensorRT 11 and 10.16 engines to be byte-identical. The
diagnostics exporter requires four valid Madrid `trtexec` suites, the matching
clean environment, a valid Nsight output contract, and retained trace/SQLite
evidence; overlap and copy findings are recomputed from SQLite.

Asset preparation, runners, quality gates, and aggregation execute in Docker.
The goal coordinator runs on the host and requires Python `>=3.10,<3.13`.
Override its executable when needed:

```bash
HOST_PYTHON=/usr/bin/python3.12 \
  benchmarks/bin/run-benchmark.sh comparative
```

## Workflows

Choose the intended result; the coordinator builds the required images,
prepares and verifies assets, builds every selected engine on the current GPU,
runs smoke checks, and executes the required measurement stages:

```bash
benchmarks/bin/run-benchmark.sh project
benchmarks/bin/run-benchmark.sh comparative
benchmarks/bin/run-benchmark.sh tuned
benchmarks/bin/run-benchmark.sh diagnostics
```

With no filters, each goal covers RealESRGAN and SPAN at both 720p and 1080p.
Scope a run when the complete matrix is not required:

```bash
benchmarks/bin/run-benchmark.sh comparative \
  --workload span \
  --variant 1080p
```

`--dry-run` prints the complete ordered command plan without Docker or GPU
access. `--resume` continues the exact revision, matrix, profile, and selection
recorded under `artefacts/benchmarks/workflows/`; it never guesses completion
from arbitrary files. Run without `--resume` refuses to overwrite existing
state.

The goals are intentionally separate:

- `project` measures only `trtvideo` for regression work;
- `comparative` runs quality gates and rotated project/vstrt/VSGAN campaigns
  using pinned upstream defaults;
- `tuned` runs adaptive searches, winner quality gates, final campaigns, and
  cross-resolution publication checks;
- `diagnostics` records `trtexec` ceilings for the selection and the canonical
  SPAN 1080p Nsight trace when that combination is selected.

All workflows reuse the same runners and validation contracts. Raw artifacts
remain separated under `artefacts/benchmarks/project/`,
`artefacts/benchmarks/comparative/` (including tuned evidence), and
`artefacts/benchmarks/diagnostics/`. The Make targets documented below are the
low-level troubleshooting interface, not the normal full-cycle workflow.

## Comparison Matrix

- `run-vstrt` - pinned vstrt with a selectable scheduling profile and the same
  TensorRT 11 engine.
- `run-vsgan` - pinned upstream VSGAN with a selectable scheduling profile and
  a separate TRT10.16 engine because serialized engines are incompatible across
  runtime versions.
- `run-trtexec` - diagnostic inference ceiling, not a competitor.
- `profile-nsight` - one non-publishable project timeline for pipeline analysis.
- `tensor-quality` - validate TensorRT outputs from exact shared FP32 RGB inputs
  and record production-preprocessing differences outside the timed path.
- `product-output-parity` - retain one canonical MP4 per product, run complete
  PSNR/SSIM decode comparisons, and generate visual crops.
- `quality-gates` - run tensor quality and decoded product-output quality.
- `run-comparative` - canonical rotation of project/vstrt/VSGAN by round and
  generation of a shared acceptance table.

`run-trtexec` stores each suite under
`artefacts/benchmarks/diagnostics/trtexec/<workload>-<variant>/`, preventing
results for different models at the same resolution from overwriting each
other.

`profile-nsight` wraps one ordinary unprofiled `trtvideo` process. It does not
contribute FPS values to a campaign. The canonical diagnostic trace uses SPAN
at 1080p because its lighter inference makes pipeline gaps more visible:

```bash
make -C benchmarks plan-nsight \
  MANIFEST=benchmarks/workloads/liveaction_span_madrid.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine

make -C benchmarks profile-nsight \
  MANIFEST=benchmarks/workloads/liveaction_span_madrid.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
```

The 120-frame trace and CLI reports are written under
`artefacts/benchmarks/diagnostics/nsight/liveaction_span_madrid-1080p/`.
`manifest.json` records the workload, engine and image contract; the ignored
`.nsys-rep` can be opened with a matching or newer Nsight Systems GUI. The
runner requires GPU video tracing and validates the complete output after
collection.

Video2X is excluded because it did not run the canonical
`RealESRGAN_x2plus`; its FPS therefore did not answer the same-model performance
question.

## Execution Profiles

`EXECUTION_PROFILE` selects the scheduling contract for the complete comparison:

- `upstream-default` is the default and uses the settings recorded from each
  pinned upstream;
- `tuned` requires explicit `--requests`, `--num-streams`, `--vs-threads`, and
  `--cuda-graph` or `--no-cuda-graph` values.

For example, these commands only generate plans and do not require a GPU:

```bash
make -C benchmarks plan-vstrt \
  EXECUTION_PROFILE=upstream-default \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine

make -C benchmarks plan-vsgan \
  EXECUTION_PROFILE=upstream-default \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

The canonical tuned workflow is manifest-driven:

```bash
make -C benchmarks run-tuned-sweep \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
make -C benchmarks run-tuned-quality \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
make -C benchmarks run-tuned-campaign \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

The canonical workflow matrix selects a predeclared adaptive tuning contract
for each workload. RealESRGAN uses `benchmarks/tuning/candidates.json`; SPAN
uses `benchmarks/tuning/span_candidates.json`. A one-run reconnaissance pass
searches streams `1..8`, applies the declared early-stop and sentinel rules, and
shortlists three candidates. A materially increasing stream-8 boundary rejects
the search and requires a wider contract. A candidate that exceeds available GPU
memory is retained as a hashed resource-ceiling artifact and excluded from
ranking; unrelated failures remain fatal. Shortlisted candidates are
independently remeasured with the full 1000-frame 3+2 contract before selection.
Only the selected pair runs exact-profile shared-input inference and
product-output gates plus the non-gating preprocessing diagnostic. A
candidate-specific inference or product-output failure disqualifies that point
and promotes the next confirmed candidate.

RealESRGAN reconnaissance uses 300 frames and records, but does not enforce,
average bitrate because NVENC CBR does not reliably converge over that short
window. This evidence is search-only and non-publishable. Confirmation,
quality, and the final campaign use 1000 frames with bitrate validation enabled.
The machine-readable `search-state.json` proves the measured points, stop
reason, sentinel or resource ceiling, shortlist, and CUDA Graph probe used by
selection.

Run the same three commands independently for 720p. A single-resolution tuned
campaign is evidence, not a publication unit. `verify-tuned-matrix` grants
publication status only when both 720p and 1080p campaigns and full quality
reports match the same workload, revision, and GPU contract.

Every profile has isolated artifact directories. A rotated campaign stores an
immutable `campaign.config.json` and rejects `RESUME=1` if the selected profile
or either runner argument string changes. For example:

```bash
make -C benchmarks run-comparative \
  EXECUTION_PROFILE=upstream-default \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

`upstream-default` is not automatically the fastest vstrt configuration:
upstream keeps one TensorRT stream and recommends increasing it when the GPU is
not saturated. The tuned workflow searches the declared `1..8` range
adaptively, confirms the strongest candidates from scratch, and records either
a proven early stop, the upper boundary, or a reproducible resource ceiling.
Manual `VSTRT_ARGS`/`VSGAN_ARGS` runs remain diagnostic and do not replace the
manifest-driven selection report.

## Assets

RealESRGAN is the default workload:

```bash
make build
make -C benchmarks prepare
make -C benchmarks verify
```

For SPAN:

```bash
make -C benchmarks prepare \
  MANIFEST=benchmarks/workloads/liveaction_span_madrid.json
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_madrid.json
```

The first run downloads approximately 168 MiB of CC0 live-action source data.
Interrupted downloads resume through HTTP range requests. Both workloads reuse
this source and the prepared clips. Model weights, generated ONNX files, and
clips remain in ignored `models/` and `videos/` directories.

Recreate only the clips without exporting the models again:

```bash
make -C benchmarks prepare ARGS=--force-clips
```

SPAN weights use the `CC-BY-NC-SA-4.0` license. The benchmark tooling does not
redistribute them and records license/attribution data in the asset lock.

## Images And Plans

```bash
make -C benchmarks build
make -C benchmarks build-vstrt
make -C benchmarks build-vsgan
```

The VSGAN wrapper uses the pinned `latest_no_avx512` image. In the corresponding
`minimal_no_avx512` release, the native `vspipe` binary was replaced by an
incompatible Python entrypoint. The benchmark runner starts from a separate
virtual environment by absolute path without activating it for the embedded
Python inside VSScript. The Docker build validates both Python environments and
the native binary.

Upstream FFmpeg requires NVENC API 13.1 and driver 610+. The benchmark wrapper
uses pinned Ubuntu FFmpeg `7:6.1.1-3ubuntu5` as an external encoder adapter and
the source of `ffprobe`; the pinned VSGAN inference stack remains unchanged.

Command-generation checks do not require a GPU. A VSGAN plan needs the path of
the future TRT10 engine, but the file itself is optional in dry-run mode:

```bash
make -C benchmarks dry-run \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine \
  VARIANT=720p \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250" \
  VSTRT_ARGS="--warmup-frames 24" \
  VSGAN_ARGS="--warmup-frames 24"
```

For SPAN, override the model paths together with `MANIFEST`:

```bash
MANIFEST=benchmarks/workloads/liveaction_span_madrid.json
ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx
ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine

make -C benchmarks dry-run \
  MANIFEST="$MANIFEST" ONNX="$ONNX" ENGINE="$ENGINE" \
  VSGAN_ENGINE="$VSGAN_ENGINE"
```

Frame/run parameters may be reduced only for smoke tests. Such a suite can be
valid, but it receives `scope: acceptance` and `publishable: false`. The same
restriction applies to a canonical individual suite: comparisons may be
published only from the shared rotated campaign. Canonical workflow smoke runs
record actual bitrate without enforcing the 10% average-bitrate threshold;
full campaigns and product-output quality runs continue to enforce it.

All video runners use one explicit NVENC contract: H.264 P4/HQ, CBR,
target=min=max bitrate, a two-second VBV buffer with 50% initial occupancy,
single pass, lookahead/AQ disabled, a one-second GOP, and zero B-frames.

Run the independent tensor-space quality job after the GPU smoke tests:

```bash
make -C benchmarks tensor-quality \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

The command captures canonical frames `0`, `499`, and `999`. It first records
each production preprocessing path as a diagnostic. It then injects the exact
trtvideo FP32 CHW RGB input tensors into TRT11 vstrt and pinned VSGAN, requires
the injected inputs to remain byte-identical, and compares only TensorRT
outputs against fixed thresholds. Raw tensors, `inference-parity.json`, and
`preprocessing-diagnostic.json` are written under
`artefacts/benchmarks/comparative/quality/model-space/<profile>/<workload>-<variant>/`.
Run the exact 720p and SPAN commands from `GPU_RUNBOOK.md`. This gate is not
included in FPS timing. When the report exists at the canonical path,
`aggregate-campaign` verifies both reports' contract version, execution profile,
workload, input, ONNX, and engine hashes plus exact image IDs and clean
repository revision. Preprocessing differences are published diagnostics and
do not fail acceptance.

Run the complete quality contract together:

```bash
make -C benchmarks quality-gates \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

The product-output job performs one separate canonical retained-output run per
implementation. It does not contribute FPS values to the rotated campaign.
`product-output-parity.json`, FFmpeg metric logs, retained MP4s, and visual PNG
crops are written under
`artefacts/benchmarks/comparative/quality/product-output/<profile>/<workload>-<variant>/`.
The fixed gate requires 1000 compared frames, average PSNR of at least 35 dB,
and overall SSIM of at least 0.95. The aggregator reloads the retained-output
run manifests and requires the same images, revision, encoder, assets, and
engines as the measured campaign. Its 1000-frame window is independent of the
workload-specific performance window.

Run the canonical campaign after smoke tests:

```bash
make -C benchmarks run-comparative \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

Results are written to
`artefacts/benchmarks/comparative/campaigns/upstream-default/realesrgan_x2plus_madrid-1080p/`:
raw manifests, `campaign.config.json`, `campaign.events.jsonl`, `campaign.json`,
and `results.md`. The config fixes scheduling identity; the event log records
the actual order, start/end time, and observed pause for each run. The aggregator
rejects either missing evidence or mixed profiles. The main table contains
median FPS, wall time, CPU cores, GPU utilization, power, VRAM, bitrate, and
size. A separate lifecycle table contains median startup, steady-state frame
loop, and finalize/mux durations, which sum to the same full-process wall time.
A stability table retains all raw FPS values and reports full spread plus an
explicit four-of-five consensus and outlier when the initial three rounds
required two additional rounds.

The production image contains neither NVML nor external benchmark tools. The
complete GPU workflow is documented in `GPU_RUNBOOK.md`.
