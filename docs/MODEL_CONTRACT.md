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

The runtime can structurally recognize any uniform integer scale, but the
current exporter, conformance contract, demo, and published evidence cover only
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

## Prepare A Source Checkpoint

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

## Start From Existing ONNX

An existing ONNX file can skip `export-onnx`. It must already represent the RGB
contract above. For a dynamic 2x graph, create a static mixed-FP16 variant:

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

## Export-Conformance Gate

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
PyTorch `pixel_unshuffle`. Successful evidence records the source checkpoint
SHA256 and size, deterministic probe identity, tool versions, metrics, and the
SHA256 and size of every exported FP32 ONNX file. The report is written as
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

## Smoke-Test The Engine

First verify the static environment:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  trtvideo:latest trtvideo doctor --disk-path /work
```

Then process a short sample whose resolution matches the engine:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  trtvideo:latest trtvideo \
  --engine /work/models/engines/model_720p.engine \
  --input /work/videos/input-720p.mp4 \
  --output /work/videos/output-1440p.mp4 \
  --max-frames 120
```

A process exit code is not enough. Fully decode the output and inspect frame
count, resolution, timestamps, pixel format, color range/space/transfer/
primaries, bitrate, and duration as described in
the [testing contract](TESTING.md).

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

The current benchmark preparation path also requires source-export conformance
for its `.pth` checkpoints. That preflight protects the model conversion path,
but the public status is grounded in the published runtime and product-output
evidence. The complete protocol for promoting a model to `validated` is the
benchmark quality contract in
[`benchmarks/methodology.md`](../benchmarks/methodology.md).
