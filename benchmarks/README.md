# Benchmarks

This directory contains reproducible workload manifests, pinned implementation
metadata, isolated Docker environments, and runners. Models, ONNX files,
TensorRT engines, source videos, and raw results are not added to Git.
Compact, privacy-reviewed publication snapshots are stored in `results/`.

The active comparison is **trtvideo / vs-mlrt / TheAnimeScripter (TAS)**.
The September RTX 3090 and RTX 4090 snapshots measure these three participants
on TensorRT 11.2.1.2. Earlier VSGAN campaigns remain in Git history, not in the
current tables. Figure generation reads each dataset's participants rather
than relabeling a column.

- `methodology.md` - execution profiles and validity criteria.
- `workloads/` - RealESRGAN and SPAN workload manifests.
- `workflows/` - canonical workload/resolution matrix for complete workflows.
- `implementations.json` - pinned implementations and execution profiles.
- `docker/` - TensorRT 11 vstrt and isolated pinned TAS environments.
- `bin/run-benchmark.sh` - goal-based host workflow entrypoint.
- `scripts/contracts/` - benchmark plans, engine metadata, and run/quality
  evidence contracts shared across runners, gates, aggregation, and tuning.
- `scripts/runtime/` - process timing, CPU/NVML sampling, environment capture,
  command execution, JSON output, media validation orchestration, and suite
  policy.
- `scripts/runners/` - product, VapourSynth, TAS, and trtexec adapters built
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
gates for the active three participants. It records selected execution
profiles, actual runtime versions, hashed run evidence, and tensor/MP4 identity
per workload without requiring independently built engines to have either
equal or different hashes. The diagnostics exporter requires four valid Madrid
`trtexec` suites, the matching
clean environment, a valid Nsight output contract, and retained trace/SQLite
evidence; overlap and copy findings are recomputed from SQLite.

New sweep figures separate the vs-mlrt stream curve from TAS's categorical
CPU/NVDEC x FFmpeg/neLux grid. Missing or disqualified configurations are not
drawn as zero-FPS measurements. Throughput/resource figures select the fastest
external result from that dataset, with its name next to the bars. Historical
VSGAN snapshots still render with their original stream-curve layout.

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
- `comparative` runs quality gates and rotated trtvideo/vstrt/TAS campaigns
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
- `run-tas` - pinned TheAnimeScripter with CPU/NVDEC decoding and FFmpeg/neLux
  output, using a separately built engine from the canonical ONNX.
- `run-trtexec` - diagnostic inference ceiling, not a competitor.
- `profile-nsight` - one non-publishable project timeline for pipeline analysis.
- `tensor-quality` - validate TensorRT outputs from exact shared FP32 RGB inputs
  and record production-preprocessing differences outside the timed path.
- `product-output-parity` - retain one canonical MP4 per product, run complete
  PSNR/SSIM decode comparisons, and generate visual crops.
- `quality-gates` - run tensor quality and decoded product-output quality.
- `run-comparative` - canonical rotation of trtvideo/vstrt/TAS by round and
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
- `tuned` requires explicit participant-specific settings. vs-mlrt uses
  `--requests`, `--num-streams`, `--vs-threads`, and `--cuda-graph` or
  `--no-cuda-graph`; TAS uses `--decode-method cpu|nvdec` and
  `--writer ffmpeg|nelux`, retaining its native CUDA Graph execution.

TAS `upstream-default` uses CPU decode and the FFmpeg writer. The common model,
precision, and NVENC contract are still normalized, so this is an I/O-default
baseline, not an unmodified factory invocation. TAS profiles contain
`execution_profile`, `decode_method`, `writer`, and `cuda_graph=true`; they do
not have synthetic VapourSynth requests, stream counts, or thread counts.

For example, these commands only generate plans and do not require a GPU:

```bash
make -C benchmarks plan-vstrt \
  EXECUTION_PROFILE=upstream-default \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine

make -C benchmarks plan-tas \
  EXECUTION_PROFILE=upstream-default \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
```

The canonical tuned workflow is manifest-driven:

```bash
make -C benchmarks preflight-tas-quality \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
make -C benchmarks run-tuned-sweep \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
make -C benchmarks run-tuned-quality \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
make -C benchmarks run-tuned-campaign \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
```

The canonical workflow matrix selects a predeclared adaptive tuning contract
for each workload. RealESRGAN uses `benchmarks/tuning/candidates.json`; SPAN
uses `benchmarks/tuning/span_candidates.json`. A one-run vs-mlrt reconnaissance
pass searches streams `1..8`, applies the declared early-stop and sentinel rules, and
shortlists three candidates. A materially increasing stream-8 boundary rejects
the search and requires a wider contract. A candidate that exceeds available GPU
memory is retained as a hashed resource-ceiling artifact and excluded from
ranking; unrelated failures remain fatal. Shortlisted candidates are
independently remeasured with the full 1000-frame 3+2 contract before selection.
TAS instead checks quality for the four CPU/NVDEC x FFmpeg/neLux combinations
before performance search, runs one reconnaissance measurement per eligible
combination, then confirms the top three (or all remaining if fewer). The
finite grid has no stream early-stop rule. Among TAS points within 1% of the
confirmed peak, the lower median peak VRAM wins, then lower CPU, then stable
candidate ID. OOM excludes only that combination; thermal slowdown remains a
failed measurement, not a tuning choice.

The selected pair then runs exact-profile shared-input inference and
product-output gates plus the non-gating preprocessing diagnostic. A
candidate-specific inference or product-output failure disqualifies that point
and promotes the next confirmed candidate.

RealESRGAN reconnaissance uses 300 frames and records, but does not enforce,
average bitrate because NVENC CBR does not reliably converge over that short
window. This evidence is search-only and non-publishable. Confirmation,
quality, and the final campaign use 1000 frames with bitrate validation enabled.
The machine-readable `search-state.json` proves the measured points,
implementation-specific completion reason, resource evidence, and shortlist;
vs-mlrt also records its sentinel and CUDA Graph probe. No TAS graph-off
configuration is invented by the benchmark adapter.

Run the same four commands independently for 720p, setting `VARIANT=720p`
and the matching engine paths on each command. A single-resolution tuned
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
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
```

`upstream-default` is not automatically the fastest vstrt configuration:
upstream keeps one TensorRT stream and recommends increasing it when the GPU is
not saturated. The tuned workflow searches the declared `1..8` range
adaptively, confirms the strongest candidates from scratch, and records either
a proven early stop, the upper boundary, or a reproducible resource ceiling.
Manual `VSTRT_ARGS`/`TAS_ARGS` runs remain diagnostic and do not replace the
manifest-driven selection report.

The lower-level tuning CLI accepts `--tas-engine`; Make exposes the same path
as `TAS_ENGINE`. The campaign CLI records `--tas-arguments`, exposed by Make
as `TAS_ARGS`. Existing VSGAN workflow state is incompatible with
the new participant contract. Preserve it in an archive and start new evidence
rather than editing its configuration to make `--resume` accept it.

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
make -C benchmarks build-tas
```

TAS runs in its own pinned Python/PyTorch/neLux/TensorRT environment; these
dependencies are not added to the trtvideo production image. The benchmark
adapter selects a prebuilt engine and normalizes both output backends to the
same NVENC contract without changing TAS's queues, inference, CUDA Graph,
color conversion, or synchronization. Backend fallback, model downloads, and
engine rebuilding during measurement are errors. TAS's internal no-output
`--benchmark` mode is not used.

This is an adapted upstream pipeline, not a completely unmodified CLI. For a
custom static ONNX, TAS's square-shape multiple-detection probe can fall back
to alignment 16 and pad a 1080p input to 1088 pixels, incompatible with the
canonical engine. The adapter instead requires processor width/height to
equal the declared engine shape and bypasses heuristic padding with multiple
1. It changes neither pixels nor the model graph. It also replaces interactive
dependency installation with the image-locked bootstrap. These normalizations,
prebuilt-engine lookup, and encoder settings are covered by adapter provenance;
the native per-frame and TensorRT operations are unchanged.

Build the TAS engine outside timing with the upstream builder:

```bash
make -C benchmarks build-tas-engine \
  VARIANT=720p \
  ONNX=models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_720p_fp16.onnx \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_720p.engine
```

The graph is already mixed-FP16 with FP32 boundaries. TAS uses `--half false`
to retain those boundaries, not to re-export or promote the entire graph.

Command-generation checks do not require a GPU. The future TAS engine path
must be supplied, but the file itself is optional in dry-run mode:

```bash
make -C benchmarks dry-run \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_720p.engine \
  VARIANT=720p \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250" \
  VSTRT_ARGS="--warmup-frames 24" \
  TAS_ARGS="--warmup-frames 24"
```

For SPAN, override the model paths together with `MANIFEST`:

```bash
MANIFEST=benchmarks/workloads/liveaction_span_madrid.json
ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx
ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
TAS_ENGINE=models/benchmarks/liveaction-span/engines/tas/liveaction_span_1080p.engine

make -C benchmarks dry-run \
  MANIFEST="$MANIFEST" ONNX="$ONNX" ENGINE="$ENGINE" \
  TAS_ENGINE="$TAS_ENGINE"
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

Asset preparation also runs a model export-conformance preflight once per
checkpoint. A deterministic 16x16 RGB probe must match between the original
FP32 PyTorch model and FP32 ONNX before engines are built. The JSON evidence is
bound to the inferred scale, source, toolchain, export-contract, and full-size
FP32 ONNX hashes and is summarized in the asset lock. It is not repeated per
resolution candidate or included in benchmark timing.

Run the independent tensor-space quality job after the GPU smoke tests:

```bash
make -C benchmarks tensor-quality \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
```

The command captures canonical frames `0`, `499`, and `999`. It first records
each production preprocessing path as a diagnostic. It then injects the exact
trtvideo FP32 CHW RGB input tensors into TRT11 vstrt and the native TAS TensorRT
class, requires
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
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
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
  TAS_ENGINE=models/benchmarks/realesrgan-x2plus/engines/tas/realesrgan_x2plus_1080p.engine
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
