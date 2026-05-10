# AI Video Upscaler

CLI tools for AI video upscaling via TensorRT. Supports RealESRGAN and SPAN models in
`.pth` and ONNX formats.

The recommended workflow is Docker. The container includes the runtime dependencies for
TensorRT inference, NVDEC/NVENC inference, ONNX preparation, and model export.

Tested on Tesla T4.

## Structure

```
inference.py          - video upscale via ffmpeg pipe + TensorRT
inference_gpu.py      - video upscale via NVDEC/NVENC + TensorRT
scripts/run_batch.sh  - batch video processing
tools/export_onnx.py  - export .pth -> .onnx
tools/prepare_onnx.py - fix dynamic axes in ONNX
tools/build_engine.py - compile .onnx -> .engine
models/               - data (not in git)
  pretrained/         - .pth files
  onnx/               - .onnx files
  engines/            - .engine files
videos/               - input/output videos (not in git)
```

## Host Requirements

Docker runs that use GPU require:

- NVIDIA driver on the host
- Docker
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA runtime/CDI configured for Docker

Configure Docker after installing NVIDIA Container Toolkit:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that GPU passthrough works:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

If Docker uses CDI, `nvidia-ctk cdi list` should show devices such as
`nvidia.com/gpu=all`.

## Build

```bash
docker build -t upscaler:latest .
```

The image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, which is required for
NVDEC/NVENC through PyNvVideoCodec. Run containers with `--gpus all`.

## Docker Workflow

### 1. Export `.pth` to ONNX

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  upscaler:latest export-onnx \
  --model_path models/pretrained/RealESRGAN_x2plus.pth
```

### 2. Prepare ONNX

Use this when an ONNX model has dynamic axes and needs fixed input shapes for TensorRT.
If `--size` is omitted, the tool creates the default 1280x720 and 1920x1080 variants.

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  upscaler:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720
```

### 3. Build TensorRT Engine

Compilation takes 5-15 minutes. The engine is tied to the TensorRT version and GPU class.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  upscaler:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/engines/model_720p.engine
```

Dynamic ONNX files can also be built directly with an explicit TensorRT optimization profile:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  upscaler:latest build-engine \
  models/onnx/model.onnx \
  -o models/engines/model_dynamic_720p.engine \
  --min-shape input:1x3x360x640 \
  --opt-shape input:1x3x720x1280 \
  --max-shape input:1x3x1080x1920
```

Current video inference commands are static-shape full-frame paths. For `upscale-video` and
`upscale-video-nvcodec`, use a static ONNX variant from `prepare-onnx` and build a static
engine. Dynamic-profile engine build support is the foundation for a later dynamic runtime
path.

### 4. Upscale Video

Default video command: ffmpeg handles decode/encode, TensorRT runs on GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

NVDEC/NVENC backend: decode, color conversion, TensorRT inference, and encode stay on GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video-nvcodec \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

Select a specific CUDA device with `--gpu-id`:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video-nvcodec \
  --gpu-id 1 \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

## CLI Commands

Canonical video commands:

```bash
upscale-video           # ffmpeg decode/encode + TensorRT
upscale-video-nvcodec   # NVDEC/NVENC + TensorRT
```

Compatibility aliases are still available:

```bash
upscale      # alias for upscale-video
upscale-gpu  # alias for upscale-video-nvcodec
```

Tooling commands:

```bash
export-onnx
prepare-onnx
build-engine
```

Common inference options:

```bash
--engine PATH       Path to .engine file
--input PATH        Input video
--output PATH       Output video (default: *_upscaled.ext)
--gpu-id N          CUDA GPU index (default: 0)
--max-frames N      Limit frames (0 = all)
--profile           Print per-stage profiling
--verbose           Verbose output
--quiet             Minimal output
```

Backend-specific options:

```bash
upscale-video:         --crf N
upscale-video-nvcodec: --crf N --codec h264|hevc
```

Note: `--crf` in the NVDEC/NVENC backend is mapped to an estimated NVENC bitrate. It is
not identical to x264 CRF.

## Docker Compose

ffmpeg backend:

```bash
docker compose run --rm upscale-video
```

NVDEC/NVENC backend:

```bash
docker compose run --rm upscale-video-nvcodec
```

Edit `docker-compose.yml` to set the engine and input paths.

## Developer Install

Use source installs only for development. Docker is the supported workflow for normal
runs.

```bash
pip install -e ".[ffmpeg]"
pip install -e ".[gpu]"
pip install -e ".[export]"
pip install -e ".[dev]"
```

## Quality Checks

Local checks:

```bash
ruff check .
mypy .
python3 -m compileall -q inference.py inference_gpu.py tools upscaler
```

Docker-based checks:

```bash
docker build --build-arg INSTALL_DEV=1 -t upscaler:dev .
docker run --rm -v "$PWD:/app" upscaler:dev ruff check .
docker run --rm -v "$PWD:/app" upscaler:dev mypy .
docker run --rm -v "$PWD:/app" upscaler:dev \
  python3 -m compileall -q inference.py inference_gpu.py tools upscaler
```
