# AI Media Enhancer

CLI tools for TensorRT-based AI media processing. Video upscaling is the
currently implemented workflow. Model preparation supports `.pth` checkpoints
and existing ONNX files; inference runs with an explicitly selected TensorRT
engine.

Docker is the recommended workflow. The image contains runtime dependencies for
TensorRT inference, NVDEC/NVENC inference, ONNX preparation, and model export.

The production runtime currently uses Python 3.12 from the
`nvcr.io/nvidia/tensorrt:26.06-py3` base TensorRT Docker image.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - inference, TensorRT runtime, and backend
  architecture.
- [Testing](docs/TESTING.md) - test layers and the Docker-only quality gate.
- [Roadmap](docs/ROADMAP.md) - current development directions.
- [Changes](docs/CHANGES.md) - versioned changes and versioning rules.
- [Performance Log](docs/PERFORMANCE_LOG.md) - measured performance changes.
- [Benchmark Methodology](benchmarks/methodology.md) - workloads and comparison
  rules.
- [Published Benchmark Results](benchmarks/results/README.md) - validated
  competitor baselines and machine-readable snapshots.

## Host Requirements

GPU runs require a host with the following components already configured:

- an NVIDIA driver;
- Docker;
- GPU passthrough for `docker run --gpus all`.

## Build The Image

```bash
make build
```

The default image is `ai-media-enhancer:latest`. Use
`make build IMAGE=example/name:tag` to select another name.

The image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, which is
required for NVDEC/NVENC through PyNvVideoCodec. Containers run with
`--gpus all`.

Build the development image with Ruff and mypy:

```bash
make build-dev
```

## Models

Model weights, ONNX files, and TensorRT engines are not included in the
repository. The recommended local structure is:

```text
models/
  pretrained/   # source .pth checkpoints
  onnx/         # source and prepared .onnx files
  engines/      # TensorRT .engine files and .engine.json sidecars
  cache/        # TensorRT timing cache
```

This structure is a convention used by the examples, not a CLI restriction. Any
path available inside the container may be used. With
`-v "$PWD/models:/app/models"`, the local `./models` directory is available to
commands as `models/`.

`export-onnx` loads compatible image-to-image `.pth` checkpoints through
Spandrel. The current exporter creates 720p and 1080p variants for 2x upscaling
and has been verified with RealESRGAN_x2plus. An existing ONNX file can be
passed directly to `prepare-onnx`.

## Docker Workflow

### 1. Export `.pth` To ONNX

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest export-onnx \
  --model_path models/pretrained/RealESRGAN_x2plus.pth
```

### 2. Prepare ONNX

Use this step when an ONNX model has dynamic axes and TensorRT requires fixed
input shapes. Without `--size`, the command creates default variants for
1280x720 and 1920x1080.

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720
```

With TensorRT 11, FP16 is defined by tensor types inside ONNX instead of a
builder flag. Create the mixed-precision variant during this step:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720 \
  --precision fp16
```

`--precision fp16` performs a lightweight ONNX graph rewrite through
`onnxconverter-common`: internal floating-point tensors are converted to FP16,
while input/output tensors remain FP32 to preserve the current video runtime
contract. This step does not require a GPU.

### 3. Build A TensorRT Engine

Compilation time depends on the model and GPU. An engine is tied to the
TensorRT version and GPU class.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/engines/model_720p.engine \
  --timing-cache models/cache/trt.cache
```

`build-engine` automatically creates a sidecar manifest next to the engine:

```text
models/engines/model_720p.engine.json
```

The sidecar stores the ONNX hash, engine hash, TensorRT version, precision,
input/output shapes, profile, and builder flags. Use `--manifest PATH` to select
another location, or `--no-manifest` to disable it.

An FP16 engine is built from an FP16/mixed-precision ONNX without additional
precision flags in `build-engine`:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p_fp16.onnx \
  -o models/engines/model_720p_fp16.engine \
  --timing-cache models/cache/trt.cache
```

TensorRT 11 removed weak-typing flags such as `BuilderFlag.FP16`. To use FP16,
first create an ONNX file with `prepare-onnx --precision fp16`, then pass that
file to `build-engine`.

#### Dynamic Engine: Build Only

A dynamic ONNX can be built directly when an explicit TensorRT optimization
profile is provided:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model.onnx \
  -o models/engines/model_dynamic_720p.engine \
  --min-shape input:1x3x360x640 \
  --opt-shape input:1x3x720x1280 \
  --max-shape input:1x3x1080x1920 \
  --timing-cache models/cache/trt.cache
```

The resulting dynamic engine cannot be used by the current video inference
runtime. `upscale --backend ffmpeg|nvcodec` requires a static ONNX variant from
`prepare-onnx` and a corresponding static engine.

### 4. Upscale Video

`--engine` is required and must match the input video resolution. The runtime
checks the engine input shape and exits on a mismatch.

With the FFmpeg backend, FFmpeg performs decode and encode while TensorRT
inference runs on the GPU:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend ffmpeg \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

With the NVDEC/NVENC backend, decode, color conversion, TensorRT inference, and
encode remain on the GPU:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend nvcodec \
  --engine models/engines/model_720p.engine \
  --bitrate-mbps 35 \
  --input videos/input.mp4
```

Both backends replace the source video with the enhanced stream and copy every
source audio, subtitle, data, and attachment stream without transcoding. Global
metadata and chapters are retained. Before TensorRT is initialized, a short
FFmpeg preflight verifies that all copied streams are supported by the selected
output container. For example, an MP4 output is rejected when an SRT subtitle or
attachment cannot be represented; choose an `.mkv` output in that case.

Output is written to a same-directory temporary file and replaces the requested
path atomically only after decode, encode, and mux complete successfully. A
failed run returns a non-zero status, removes the partial temporary file, and
leaves an existing output untouched.

With `--max-frames`, copied streams are shortened to the processed video
duration. Chapters are omitted because their original timestamps may point
beyond the shortened output.

Select a specific CUDA device:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend nvcodec \
  --gpu-id 1 \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

### 5. Prepare A Benchmark Workload

The canonical workload uses `RealESRGAN_x2plus` and the public lossless Sintel
trailer. The command downloads approximately 3.7 GB of source data, creates H.264
inputs for 720p/1080p, and exports static mixed-FP16 ONNX files. It does not build
a TensorRT engine; engines must be built on the GPU used for the benchmark.

```bash
make -C benchmarks prepare
make -C benchmarks verify
```

Models and videos remain in ignored `models/` and `videos/` directories. Pinned
sources, checksums, license attribution, and the complete measurement contract
are documented in
[Benchmark Methodology](benchmarks/methodology.md).

### 6. Benchmark

Benchmark tools and the NVML dependency are excluded from the production image.
Build the separate image first, then run the benchmark after building the engine
on the selected benchmark GPU:

```bash
make -C benchmarks build
```

Canonical run:

```bash
make -C benchmarks run-ai-media \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine
```

The runner uses a separate 100-frame warmup process followed by at least three
measured processes of 1000 frames each. If FPS spread exceeds 5%, it runs two
additional processes. The regular benchmark does not enable `--profile`; it
measures wall time from child `upscale` startup through encode, flush, mux, and
process exit.

Run a short infrastructure check before the full benchmark:

```bash
make -C benchmarks run-ai-media \
  VARIANT=720p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  ARGS="--runs 1 --extra-runs 0 --frames 120 --warmup-frames 24 --idle-seconds 0"
```

Artifacts are written to `artefacts/benchmarks/`: suite/run manifests, child
logs, and raw NVML samples. A manifest includes end-to-end FPS, lifecycle and CPU
metrics, peak VRAM, average/peak power, energy and joules/frame, hashes, and a
sanitized environment. After SHA256 and complete FFmpeg validation, valid MP4
files are deleted while invalid output is retained.

`benchmark-upscale` can also be called directly for an arbitrary video-only
input. The `nvcodec` path requires an explicit bitrate, one backend, and a
separate output directory:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/artefacts:/app/artefacts" \
  ai-media-enhancer:benchmark benchmark-upscale \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4 \
  --backend nvcodec \
  --bitrate-mbps 35 \
  --output-dir artefacts/manual-run \
  --json -
```

Progress is written to `stderr`, so `--json -` keeps `stdout` suitable for
machine-readable JSON. Per-stage timings are collected separately with
`upscale --profile` or `upscale --profile-json` and are not treated as the
end-to-end benchmark.

The isolated runners in `benchmarks/` separate TensorRT 11
`vs-mlrt/vstrt` technical parity, stock `VSGAN-tensorrt-docker` product
comparison, and the diagnostic `trtexec` ceiling. The complete GPU-host
preparation and smoke sequence is documented in
[GPU Benchmark Runbook](benchmarks/GPU_RUNBOOK.md).

## CLI Reference

Available commands:

```bash
upscale
export-onnx
prepare-onnx
build-engine
benchmark-upscale
```

Use `--help` to view all arguments:

```bash
docker run --rm ai-media-enhancer:latest upscale --help
docker run --rm ai-media-enhancer:benchmark benchmark-upscale --help
docker run --rm ai-media-enhancer:latest export-onnx --help
docker run --rm ai-media-enhancer:latest prepare-onnx --help
docker run --rm ai-media-enhancer:latest build-engine --help
```

## Encoding

The FFmpeg backend uses `libx264` and controls quality through `--crf`, which
defaults to 18. The NVDEC/NVENC backend does not support `--crf`; select the
codec with `--codec h264|hevc`.

Without `--bitrate-mbps`, the NVDEC/NVENC backend estimates target bitrate from
the source video bitrate:

```text
source_bitrate * (pixel_ratio * fps_ratio) ** 0.6
```

This reduces the risk of unexpectedly large output after upscaling. Use an
explicit `--bitrate-mbps` for fully controlled output size.

If `ffprobe` cannot determine the source bitrate, the NVDEC/NVENC backend
requires an explicit `--bitrate-mbps` and exits with an error. `--crf` is
supported only by the FFmpeg backend.

## Media Contract

The current media contract targets SDR 8-bit video. The `nvcodec` backend fails
fast for inputs other than `yuv420p`/`nv12` and for HDR transfer functions.
NV12/RGB conversion uses an explicit CV-CUDA color specification, and the output
receives explicit color tags instead of `unknown`.

## Docker Compose

FFmpeg backend:

```bash
docker compose run --rm upscale-ffmpeg
```

NVDEC/NVENC backend:

```bash
docker compose run --rm upscale-nvcodec
```

Engine and input video paths are configured in `docker-compose.yml`.

## Quality Checks

Checks run through the Docker development image. Unit tests do not require a GPU
and must not import TensorRT, CV-CUDA, or PyNvVideoCodec.

Docker-based checks:

```bash
make build-dev
make check
```

The test-layer architecture is documented in `docs/TESTING.md`.
