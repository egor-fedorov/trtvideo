# Model Contract

This document defines the model boundary supported by the current production
video runtime. It separates structural compatibility, source-export
conformance, and published validation. Passing one layer does not imply that
the later layers pass.

## Runtime Contract

The production runtime accepts a TensorRT engine with this contract:

| Property | Requirement |
|---|---|
| Task | Per-frame spatial super-resolution; no temporal state |
| I/O count | Exactly one input and one output |
| Layout | RGB NCHW |
| Shape | `[1, 3, H, W] -> [1, 3, scale*H, scale*W]` |
| Batch | Static batch 1 |
| Spatial dimensions | Static, positive, and matched to the decoded input |
| Scale | Uniform positive integer in both spatial dimensions |
| Binding dtype | FP32 or FP16 |
| Value range | Normalized RGB in `[0, 1]` |
| Execution | Full-frame only; no tiling |

The runtime, exporter, and conformance contract recognize a uniform integer
scale from tensor shapes. The demo and published evidence currently cover only
2x models. Other scales are therefore `untested`, not supported claims.

One static engine is required for each input resolution. `build-engine` can
compile a dynamic ONNX graph with an optimization profile, but the production
runtime does not select dynamic shapes or resize its buffers. A dynamic engine
is not currently usable for video processing.

TensorRT binding dtype and internal graph precision are separate. The runtime
accepts FP32 or FP16 bindings. The tested preparation path uses FP32 input and
output bindings with mixed FP16 tensors inside the ONNX graph:

```text
FP32 RGB input -> mixed-FP16 ONNX graph -> FP32 RGB output
```

`prepare-onnx --precision fp16` creates that graph and keeps I/O types unchanged.
`build-engine` compiles the types encoded in ONNX; it does not apply a separate
FP16 builder flag.

The model never receives YUV code values. The production pipeline decodes
NV12, expands limited-range input when required, converts it to RGB, normalizes
it to `[0, 1]`, and writes NCHW tensors. Output is converted back through the
declared video color contract. A model should not implement its own limited-
range expansion or YUV conversion.

## One-Command Compatibility Check

Use the published model-tools image for the normal community-report path. An
input clip is optional: by default the command downloads, hash-checks, and
prepares the same pinned Jacqueville live-action source used by the demo.

```bash
IMAGE=ghcr.io/egor-fedorov/trtvideo-model-tools:vX.Y.Z

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/work" \
  --workdir /work \
  "$IMAGE" trtvideo compatibility-check \
  --checkpoint models/pretrained/model.pth \
  --model-name 2xExampleModel \
  --model-source https://example.org/models/2xExampleModel \
  --model-license Apache-2.0 \
  --output-dir compatibility-report
```

The bind mount is also the container working directory, and generated paths are
deliberately relative to it. The terminal output and `commands.txt` therefore
refer to the same paths from the host directory after the container exits,
without embedding a contributor-specific absolute home directory.

The immutable image identity is embedded at release build time; no
`TRTVIDEO_IMAGE_REF` override is needed. The command announces a 10-30 minute
model/GPU-dependent estimate, streams child output, prints a fixed `N/total`
step bar, and emits an append-only heartbeat every 30 seconds. It writes exact
commands automatically and retains the prepared input, ONNX, conformance
evidence, engine and sidecar, timing cache, processed MP4, JSON report, and
issue-ready Markdown.

After reviewing the generated Markdown, submit it without re-entering the
evidence into the browser form:

```bash
gh issue create \
  --repo egor-fedorov/trtvideo \
  --title "[Model]: 2xExampleModel" \
  --label "model compatibility" \
  --body-file compatibility-report/model-compatibility-issue.md
```

`--dry-run` prints the complete plan without creating the output directory,
downloading media, running `doctor`, or using the GPU. `--resume` always reruns
`doctor`, then accepts the journal only when the model/input identities,
parameters, image revision, and GPU/driver/TensorRT fingerprint match. Generated
files are hash-checked; changing one invalidates that step and all downstream
steps. A failed final validation retains its diagnostic JSON and Markdown;
`--resume` replaces them while retrying that unfinished tail. User-supplied
checkpoints, ONNX files, and input videos are never deleted.

For an existing ONNX graph, replace `--checkpoint` with `--onnx`. A fully static
graph supplies its own input resolution and is passed directly to
`build-engine`, and this route works in the narrower production image. A dynamic
graph is prepared as mixed FP16 in model-tools and requires `--scale N` unless
scale metadata proves the value. A custom `--input` must be SDR BT.709, is
normalized to the media contract at its native resolution, and must match a
static ONNX input shape. HDR and non-BT.709 inputs are rejected rather than
silently relabeled.

The production image rejects checkpoint and dynamic-ONNX compatibility checks
with a precise model-tools instruction. Static ONNX needs no PyTorch, Spandrel,
ONNX Runtime, or graph conversion and therefore remains available there.

For the static-ONNX route, change the image and source argument while retaining
the other metadata and mount arguments from the command above:

```bash
IMAGE=ghcr.io/egor-fedorov/trtvideo:vX.Y.Z

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  --workdir /work \
  "$IMAGE" trtvideo compatibility-check \
  --onnx models/onnx/model.onnx \
  --model-name 2xExampleModel \
  --model-source https://example.org/models/2xExampleModel \
  --model-license Apache-2.0 \
  --output-dir compatibility-report
```

## Manual Low-Level Workflow

### Prepare The Compatibility Input

The default source does not need to be found or downloaded manually. This
low-level command downloads the pinned Jacqueville source, verifies its size and
SHA256, creates a 120-frame H.264/AAC SDR input, fully decodes it, checks its
timestamps and media metadata, and writes a hash-bound attribution manifest:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  --workdir /work \
  trtvideo:latest prepare-compatibility-input \
  --output videos/compatibility-input.mp4 \
  --manifest videos/compatibility-input.json
```

Add `--input videos/custom.mp4 --size WIDTHxHEIGHT` to normalize a custom
SDR BT.709 source instead. The command never overwrites or deletes that source.

### Prepare A Source Checkpoint

`export-onnx` accepts an image-to-image `.pth` checkpoint recognized by
Spandrel. Build both local images first:

```bash
make build-model-tools
make build
```

Export one static FP32 ONNX variant and its source-conformance report:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  trtvideo:model-tools export-onnx \
  --model_path models/pretrained/model.pth \
  --output_dir models/onnx \
  --name model \
  --size 1280x720
```

Create the tested mixed-FP16 graph while retaining FP32 bindings:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  trtvideo:model-tools prepare-onnx \
  models/onnx/model_720p.onnx \
  --output_dir models/onnx \
  --precision fp16
```

Compile the static engine on the target GPU:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  trtvideo:latest build-engine \
  models/onnx/model_720p_fp16.onnx \
  --output models/engines/model_720p.engine \
  --timing-cache models/cache/trt.cache
```

TensorRT engines are tied to the TensorRT runtime and GPU class. Rebuild the
engine after moving to an incompatible image or GPU. Keep the generated
`.engine.json` sidecar with compatibility evidence, but do not commit the engine
or model weights.

### Start From Existing ONNX

An existing ONNX file can skip `export-onnx`. It must already represent the RGB
contract above. `prepare-onnx` cannot infer scale from symbolic output shapes,
so pass it explicitly for a dynamic graph. For example, prepare a dynamic 2x
graph as follows:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  trtvideo:model-tools prepare-onnx \
  models/onnx/model.onnx \
  --output_dir models/onnx \
  --size 1280x720 \
  --scale 2 \
  --precision fp16
```

For a static graph, `prepare-onnx --precision fp16` only rewrites precision. If
the graph is already static with the intended precision and bindings, pass it
directly to `build-engine`.

This route cannot prove equivalence to a source checkpoint because there is no
PyTorch reference. `prepare-onnx` changes shape metadata and precision; it does
not certify the model's RGB semantics. A successful build and video smoke test
may support a `community-reported` entry, but they do not by themselves qualify
the model as `validated`. Promotion would require a publication protocol that
also records the existing ONNX model's provenance.

### Export-Conformance Gate

Before full-size export, `export-onnx` creates a deterministic normalized RGB
probe with shape `[1, 3, 16, 16]`. It runs the probe through:

1. the original source checkpoint in PyTorch FP32;
2. an FP32 ONNX graph in ONNX Runtime CPU.

The gate requires matching output shape, finite values, and all of these
numerical thresholds:

| Metric | Requirement |
|---|---:|
| Maximum absolute error | `<= 0.0001` |
| RMSE | `<= 0.00001` |
| PSNR | `>= 80 dB` |

The exporter also rejects the incompatible `SpaceToDepth` lowering for
PyTorch `pixel_unshuffle`. It infers the uniform integer scale from the source
probe and requires every requested full-size ONNX graph to preserve that scale.
Successful evidence records the scale, source checkpoint SHA256 and size,
deterministic probe identity, tool versions, metrics, and the SHA256 and size of
every exported FP32 ONNX file. The report is written as
`NAME.export-conformance.json` only after all requested exports succeed.

Benchmark asset verification treats the report as a cache contract. A changed
checkpoint, exporter toolchain, probe contract, or ONNX file invalidates it and
requires a fresh export. The conformance step is CPU-only and is never included
in video benchmark timing.

Common failures should be interpreted as follows:

| Failure | Meaning and action |
|---|---|
| `Expected an image-to-image model` | Spandrel did not load a compatible image model. Confirm the checkpoint format and Spandrel support. |
| `SpaceToDepth` or `pixel_unshuffle` error | The ONNX lowering changes channel order. Do not bypass it; fix or extend the exporter. |
| `Export probe shape mismatch` | ONNX and the source model disagree on output shape. Inspect export lowering and model scale. |
| `non-finite values` | The source or exported graph produced NaN/Inf for the deterministic probe. |
| `max_abs`, `RMSE`, or `PSNR` failure | The exported graph is not numerically equivalent under the declared threshold. Inspect unsupported or incorrectly lowered operations. |
| `source model identity`, `tool versions`, or `ONNX identities do not match` | Cached evidence is stale. Re-run `export-onnx` with the current checkpoint and image. |
| TensorRT `ONNX Parse Error` | ONNX Runtime accepted the graph, but TensorRT does not support or parse it. This is an engine-build failure after export conformance. |

Do not relax thresholds only to make a model pass. A model-specific tolerance
requires a reviewed contract change and independent output-quality evidence.

### Smoke-Test The Engine

First verify the static environment:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  --workdir /work \
  trtvideo:latest trtvideo doctor --disk-path .
```

Then process a short sample whose resolution matches the engine:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  --workdir /work \
  trtvideo:latest trtvideo \
  --engine models/engines/model_720p.engine \
  --input videos/input-720p.mp4 \
  --output videos/output-1440p.mp4 \
  --max-frames 120
```

A process exit code is not enough. Fully decode the output and inspect frame
count, resolution, timestamps, pixel format, color range/space/transfer/
primaries, bitrate, and duration as described in
the [testing contract](TESTING.md).

### Build A Submission Bundle Manually

The one-command workflow above is preferred. For diagnosis or a nonstandard
workflow, keep the exact export, preparation, engine-build, and smoke-test
commands in a UTF-8 text file as they are run. After the output exists, one
production-image command checks the existing evidence, runs `doctor`, fully
decodes and probes the output, removes local paths from structured evidence,
and writes both JSON and an issue-ready Markdown body:

```bash
IMAGE=ghcr.io/egor-fedorov/trtvideo:vX.Y.Z

docker pull "$IMAGE"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e TRTVIDEO_IMAGE_REF="$IMAGE" \
  -v "$PWD:/work" \
  --workdir /work \
  "$IMAGE" \
  trtvideo compatibility-report \
  --model-name 2xExampleModel \
  --model-source https://example.org/models/2xExampleModel \
  --model-license Apache-2.0 \
  --source-format checkpoint \
  --source-artifact models/pretrained/model.pth \
  --export-conformance models/onnx/model.export-conformance.json \
  --engine models/engines/model_720p.engine \
  --input videos/compatibility-input.mp4 \
  --input-manifest videos/compatibility-input.json \
  --processed-output videos/output-1440p.mp4 \
  --expected-frames 120 \
  --commands-file compatibility-commands.txt \
  --output-dir compatibility-report
```

Replace `vX.Y.Z` with a published immutable release tag or use the published
image digest. Do not use `latest` for compatibility evidence.

For an existing ONNX model, use `--source-format onnx` and omit
`--export-conformance`. The report records that source-checkpoint equivalence is
unavailable without treating it as a failed community report.

`--input-manifest` is optional for manually prepared media. When supplied, the
report verifies that it hashes the actual input and carries the pinned fixture's
license attribution; `compatibility-check` always supplies it.

The command exits with status 2 when evidence is incomplete or inconsistent but
still writes both files for diagnosis. Review the Markdown before publishing it;
the command rejects common credentials, host home paths, GPU UUIDs, and SSH host
identities, but automated screening cannot prove that arbitrary command text is
safe. Submit the valid body from the host with:

```bash
gh issue create \
  --repo egor-fedorov/trtvideo \
  --title "[Model]: 2xExampleModel" \
  --label "model compatibility" \
  --body-file compatibility-report/model-compatibility-issue.md
```

## Compatibility Status

The public matrix in [`README.md`](../README.md#compatibility-matrix) uses only
these statuses:

- `validated`: a privacy-reviewed published run binds model and engine
  identities and passes shared-input TensorRT inference and decoded
  product-output quality gates;
- `community-reported`: a public issue contains enough immutable identity,
  environment, command, engine-sidecar, and output evidence to reproduce a
  successful run, but the publication protocol has not been completed;
- `untested`: no accepted evidence exists.

Submit successful and failed experiments through the
[model compatibility form](https://github.com/egor-fedorov/trtvideo/issues/new?template=model_compatibility.yml).
An issue must not contain proprietary weights, private media, credentials,
hostnames, absolute host paths, or GPU UUIDs. A maintainer reviews the evidence
before changing the matrix; opening an issue does not assign a status
automatically.

The reporter does not need to open a second pull request. The recommended
`compatibility-report` command produces the complete issue body without manual
JSON reformatting. After a successful review, the maintainer applies the
`community-reported` label and opens the focused matrix update with the issue as
its evidence link. Failed or incomplete reports remain useful compatibility
evidence but do not create a matrix row. The complete triage procedure is
documented in [`CONTRIBUTING.md`](../CONTRIBUTING.md#report-model-compatibility).

The current benchmark preparation path also requires source-export conformance
for its `.pth` checkpoints. That preflight protects the model conversion path,
but the public status is grounded in the published runtime and product-output
evidence. The complete protocol for promoting a model to `validated` is the
benchmark quality contract in
[`benchmarks/methodology.md`](../benchmarks/methodology.md).
