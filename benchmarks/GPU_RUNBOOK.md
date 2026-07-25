# GPU Benchmark Runbook

This runbook is intended for acceptance runners on one physical RTX 3090.
Rotated campaigns, exact rate control, CPU accounting, and lifecycle timing are
implemented. A publishable result still requires the quality-parity gates from
`methodology.md`. Run all commands from the repository root.

The host requires Docker, GNU Make, Git, and Python `>=3.10,<3.13`. Python is
used only by the rotated campaign coordinator; measured workloads and result
aggregation run in pinned Docker images. Set `HOST_PYTHON=/path/to/python` on
`run-comparative` when `python3` is not the intended interpreter.

Choose the workflow before following the numbered sections:

| Goal | Required sections |
|---|---|
| Project-only regression with `run-project` | Project image/assets/engine, project smoke, then Section 6 |
| Publishable competitor comparison with `run-comparative` | Sections 1-5 and 7; quality gates are mandatory |
| TensorRT or pipeline diagnosis | Relevant setup plus Section 8 or 9 |

For `run-project`, do not build vstrt/VSGAN images or engines and do not run
`quality-gates`. The project runner already performs full output decode and
validates the media, timestamp, color, and bitrate contract. Quality gates are
cross-product parity checks and are consumed only by comparative campaigns.

All comparative commands in this runbook use `VAPOURSYNTH_MODE=parity`, which
is the default. This reproduces the published one-request/one-stream baseline.
`upstream-default` and `tuned` use separate quality and campaign directories.
For tuned runs, freeze both `VSTRT_ARGS` and `VSGAN_ARGS`; resume rejects any
change to the selected profile or these argument strings.

## 1. Build And Assets

```bash
make build
make -C benchmarks build
make -C benchmarks build-vstrt
make -C benchmarks build-vsgan

make -C benchmarks prepare
make -C benchmarks verify

make -C benchmarks prepare \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
```

`prepare` does not use a GPU. RealESRGAN and SPAN reuse the same Sintel source
and clips but have separate ONNX directories.

`build-vsgan` downloads the pinned full `latest_no_avx512` image, which is
approximately 13 GB. It is used instead of the broken `minimal_no_avx512` image,
which lacks a working native `vspipe`. The wrapper installs pinned Ubuntu FFmpeg
6.1.1 for compatible NVENC encoding and output validation because upstream
FFmpeg requires driver 610+.

## 2. TensorRT Engines

Build the project's TRT11 engines with the production image on the benchmark
GPU. Build 720p and 1080p variants for both models:

```bash
mkdir -p \
  models/benchmarks/realesrgan-x2plus/engines \
  models/benchmarks/liveaction-span/engines \
  models/cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_720p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_1080p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/liveaction-span/onnx/liveaction_span_720p_fp16.onnx \
  -o models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx \
  -o models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  --timing-cache models/cache/benchmark-trt11.cache
```

Pinned VSGAN uses TensorRT 10.16 and therefore receives a separate engine built
from the same ONNX. The builder stores a log and sidecar:

```bash
make -C benchmarks build-vsgan-engine VARIANT=720p
make -C benchmarks build-vsgan-engine VARIANT=1080p

make -C benchmarks build-vsgan-engine \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_720p_fp16.onnx \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_720p.engine

make -C benchmarks build-vsgan-engine \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

Rebuild VSGAN engines after changing the pinned VSGAN base image. TensorRT
serialized plans are not compatible across different runtime builds. The runner
checks the recorded base-image digest before warmup begins.

Do not copy a serialized engine between TRT10 and TRT11. Both engines must be
built on the same benchmark GPU.

## 3. Offline Plans

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

Review the generated commands, mounted paths, and pinned implementation metadata.

## 4. GPU Smoke

Record the power limit and driver, and verify that no unrelated GPU load is
present. Then run each runner independently:

```bash
ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine
VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine
SMOKE="--frames 120 --warmup-frames 24 --runs 1 --extra-runs 0 --idle-seconds 0"

make -C benchmarks run-project \
  VARIANT=720p \
  ENGINE="$ENGINE" \
  PROJECT_OUTPUT_DIR=artefacts/benchmarks/project/smoke-realesrgan-720p \
  ARGS="$SMOKE"
make -C benchmarks run-vstrt VARIANT=720p ENGINE="$ENGINE" ARGS="$SMOKE"
make -C benchmarks run-vsgan VARIANT=720p \
  VSGAN_ENGINE="$VSGAN_ENGINE" ARGS="$SMOKE"
make -C benchmarks run-trtexec VARIANT=720p ENGINE="$ENGINE" \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250"
```

Repeat for 1080p, then for SPAN with overridden `MANIFEST`, `ENGINE`, `ONNX`, and
`VSGAN_ENGINE`. Each video runner must fully decode the output and validate the
media/timestamp contract. `trtexec` is checked separately as a diagnostic
ceiling.

## 5. Quality Gates

This section is required only for a publishable `run-comparative` campaign. Skip
it for `run-project`; running `quality-gates` always processes
`ai-media-enhancer`, vstrt, and VSGAN.

Run both independent quality jobs. The first compares model input/output FP32
RGB tensors. The second retains one canonical MP4 per product, performs complete
PSNR/SSIM decode comparisons, and generates visual crops:

```bash
make -C benchmarks quality-gates \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

Run the RealESRGAN 720p gate explicitly:

```bash
make -C benchmarks quality-gates \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine
```

Then run both resolutions for SPAN:

```bash
make -C benchmarks quality-gates \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

```bash
make -C benchmarks quality-gates \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_720p.engine
```

Each command must finish with both `Model-space parity valid` and
`Product-output parity valid`.
Inspect
`artefacts/benchmarks/comparative/quality/model-space/parity/<workload>-<variant>/model-space-parity.json`
and
`artefacts/benchmarks/comparative/quality/product-output/parity/<workload>-<variant>/product-output-parity.json`
before deleting raw tensors or retained MP4s. Review the PNG crop matrix
manually. Any threshold failure is a quality-contract failure, not benchmark
noise, and must be investigated before the campaign can be published. The
campaign aggregator automatically consumes valid reports at these canonical
paths and verifies that their profile, evidence, asset, and engine hashes match
the measured campaign. It also rejects quality evidence produced by different
Docker image IDs, a different repository revision, or a dirty build.

## 6. Project-Only Regression Benchmark

Use `run-project` for before/after measurements of `ai-media-enhancer` without
building or running competitors. It uses the same external timer, validation,
NVML sampling, warmup, and 3+2 run policy as the project row in a comparison:

```bash
make -C benchmarks run-project \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine

make -C benchmarks run-project \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine

make -C benchmarks run-project \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine

make -C benchmarks run-project \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine
```

Raw results are isolated under
`artefacts/benchmarks/project/<workload>-<variant>/`. They are regression
evidence, not a competitor comparison. Record only measured before/after
effects in `docs/PERFORMANCE_LOG.md`.

## 7. Rotated Comparative Campaign

Commit all changes and rebuild all three benchmark images before the campaign.
Preflight rejects a dirty worktree or an image not built from the current
commit.

The published RTX 3090 snapshot contains both resolutions and workloads
measured on runtime revision `0fc3037`, including media preservation. Repeat
the complete matrix only for a claim about a later runtime revision; keep the
same physical GPU, power state, commit, and image set within that campaign.

Canonical defaults are 100 warmup frames, 1000 measured frames, three rotated
rounds, and two additional rounds when the spread of any implementation exceeds
5%. Extra rounds repeat all implementations to preserve rotation and equal
sample counts. After five rounds, the campaign is valid when every
implementation either passes the full-range threshold or has a four-of-five
consensus within 5%. A consensus result is labeled `stable-with-one-outlier`;
the raw five values and all-run median remain unchanged.

```bash
make -C benchmarks run-comparative \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

For RealESRGAN 720p:

```bash
make -C benchmarks run-comparative \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine
```

For SPAN 1080p:

```bash
make -C benchmarks run-comparative \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

For SPAN 720p:

```bash
make -C benchmarks run-comparative \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_720p.engine
```

After a safe interruption, continue the same command with `RESUME=1`. Resume is
valid only when the commit, images, workload assets, and engines are unchanged.
A partial or invalid round is retained for diagnosis and requires manual
removal of only its own directory. If the process was interrupted after a run
completed but before its event was recorded, its manifest is considered
untracked and its directory must also be removed before resuming.

The campaign stores immutable `campaign.config.json`, raw manifests, append-only
`campaign.events.jsonl`, and shared `campaign.json`/`results.md` files in
`artefacts/benchmarks/comparative/campaigns/<profile>/<name>/`. The config fixes
the profile and runner arguments; the event log proves actual rotation and idle
intervals. Until the quality gates are complete, the aggregator sets
`publishable: false` even for a valid campaign. After changing only aggregation
logic, rerun `aggregate-campaign` against existing complete rounds; GPU
workloads do not need to be repeated. Keep the benchmark image used for those
measured rounds because revision validation still applies:

```bash
make -C benchmarks aggregate-campaign \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p
```

Individual `run-ai-media`, `run-vstrt`, and `run-vsgan` targets remain available
for smoke tests and diagnosis. `run-trtexec` remains a separate inference
ceiling.

Do not publish sequential execution of independent suites as the final
comparison. Raw manifests, logs, and NVML samples are not committed before
sanitization and review.

## 8. TensorRT Diagnostic Ceiling

Run the canonical `trtexec` suite separately for each TRT11 engine. It measures
inference without video decode, color conversion, encode, or mux and is used to
calculate pipeline efficiency rather than product ranking:

```bash
make -C benchmarks run-trtexec \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine
```

```bash
make -C benchmarks run-trtexec \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
```

Run both 720p ceilings explicitly:

```bash
make -C benchmarks run-trtexec \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine

make -C benchmarks run-trtexec \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine
```

Canonical defaults are 1000 measured iterations, three runs, two additional
runs when spread exceeds 5%, ten seconds between runs, and a 1000 ms `trtexec`
warmup. Results are kept separately under
`artefacts/benchmarks/diagnostics/trtexec/<workload>-<variant>/`.

The diagnostic uses the TRT11 engine shared by `ai-media-enhancer` and vstrt.
The separate TRT10 VSGAN engine is not an input to this measurement.

## 9. Nsight Systems Pipeline Diagnostic

Nsight Systems is already included in the TensorRT benchmark image. First
inspect the generated command without a GPU:

```bash
make -C benchmarks plan-nsight \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
```

On the benchmark GPU, capture the canonical 120-frame SPAN trace:

```bash
make -C benchmarks profile-nsight \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
```

Build this engine on the same physical GPU immediately before capture. A
TensorRT device-mismatch warning makes the trace diagnostic-only and requires a
clean replacement. One representative SPAN 1080p trace is sufficient; a
separate 720p Nsight capture is not required unless the 720p campaign exposes a
different bottleneck.

The runner uses the normal non-`--profile` pipeline with CUDA Graph disabled,
enables NVTX only for this subprocess, and requests CUDA, NVTX, OS-runtime,
NvVideo, and GPU video-accelerator traces. It fails if GPU video tracing is not
available or if the profiled output does not fully decode and satisfy the
120-frame media contract.

Inspect:

```text
artefacts/benchmarks/diagnostics/nsight/liveaction_span_sintel-1080p/
  ai-media.nsys-rep
  manifest.json
  stats/
  output.mp4
```

Open `ai-media.nsys-rep` in Nsight Systems 2026.3.1 or newer. Check the
`ai_media.*` NVTX ranges for per-frame H2D/D2H copies, gaps on the shared CUDA
stream, blocking CUDA/OS calls, and NVDEC/TensorRT/NVENC overlap. The CSV
reports provide command-line summaries when the GUI is unavailable. Do not use
the profiled wall time or FPS as a benchmark result.

The raw trace and generated reports remain ignored. After review, publish only
privacy-safe conclusions. Do not mark the Roadmap item complete until the trace
has been captured and interpreted.
