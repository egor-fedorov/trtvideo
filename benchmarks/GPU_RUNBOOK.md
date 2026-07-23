# GPU Benchmark Runbook

This runbook is intended for acceptance runners on one physical RTX 3090.
Rotated campaigns, exact rate control, CPU accounting, and lifecycle timing are
implemented. A publishable result still requires the quality-parity gates from
`methodology.md`. Run all commands from the repository root.

The host requires Docker, GNU Make, Git, and Python `>=3.10,<3.13`. Python is
used only by the rotated campaign coordinator; measured workloads and result
aggregation run in pinned Docker images. Set `HOST_PYTHON=/path/to/python` on
`run-campaign` when `python3` is not the intended interpreter.

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

Stock VSGAN uses TensorRT 10.16 and therefore receives a separate engine built
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

make -C benchmarks run-ai-media VARIANT=720p ENGINE="$ENGINE" ARGS="$SMOKE"
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

Run both independent quality jobs. The first compares model input/output FP32
RGB tensors. The second retains one canonical MP4 per product, performs complete
PSNR/SSIM decode comparisons, and generates visual crops:

```bash
make -C benchmarks quality-gates \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

Repeat at 720p. Then run both resolutions for SPAN:

```bash
make -C benchmarks quality-gates \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

Use the corresponding 720p engine paths for the final run. Each command must
finish with both `Model-space parity valid` and `Product-output parity valid`.
Inspect
`artefacts/benchmarks/quality/model-space-<workload>-<variant>/model-space-parity.json`
and
`artefacts/benchmarks/quality/product-output-<workload>-<variant>/product-output-parity.json`
before deleting raw tensors or retained MP4s. Review the PNG crop matrix
manually. Any threshold failure is a quality-contract failure, not benchmark
noise, and must be investigated before the campaign can be published. The
campaign aggregator automatically consumes valid reports at these canonical
paths and verifies that their evidence, asset, and engine hashes match the
measured campaign. It also rejects quality evidence produced by different
Docker image IDs, a different repository revision, or a dirty build.

## 6. Rotated Acceptance Campaign

Commit all changes and rebuild all three benchmark images before the campaign.
Preflight rejects a dirty worktree or an image not built from the current
commit.

Canonical defaults are 100 warmup frames, 1000 measured frames, three rotated
rounds, and two additional rounds when the spread of any implementation exceeds
5%:

```bash
make -C benchmarks run-campaign \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

For SPAN:

```bash
make -C benchmarks run-campaign \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

Repeat both commands with 720p paths. After a safe interruption, continue the
same campaign with `RESUME=1`. Resume is valid only when the commit, images,
workload assets, and engines are unchanged. A partial or invalid round is
retained for diagnosis and requires manual removal of only its own directory. If
the process was interrupted after a run completed but before its event was
recorded, its manifest is considered untracked and its directory must also be
removed before resuming.

The campaign stores raw manifests, append-only `campaign.events.jsonl`, and
shared `campaign.json`/`results.md` files in
`artefacts/benchmarks/campaigns/<name>/`. The event log is mandatory evidence of
actual rotation and idle intervals. Until the quality gates are complete, the
aggregator sets `publishable: false` even for a valid campaign.

Individual `run-ai-media`, `run-vstrt`, and `run-vsgan` targets remain available
for smoke tests and diagnosis. `run-trtexec` remains a separate inference
ceiling.

Do not publish sequential execution of independent suites as the final
comparison. Raw manifests, logs, and NVML samples are not committed before
sanitization and review.
