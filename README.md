# AI Video Upscaler

CLI-инструменты для AI-апскейла видео через TensorRT. Поддерживаются модели
RealESRGAN и SPAN в форматах `.pth` и ONNX.

Рекомендуемый workflow — Docker. Образ содержит runtime-зависимости для TensorRT
inference, NVDEC/NVENC inference, подготовки ONNX и экспорта моделей.

Production runtime сейчас привязан к Python 3.12 из базового TensorRT Docker image.

Проверялось на Tesla T4.

## Структура

```text
inference.py          - video upscale через ffmpeg pipe + TensorRT
inference_gpu.py      - video upscale через NVDEC/NVENC + TensorRT
scripts/run_batch.sh  - batch processing для видео
tools/export_onnx.py  - экспорт .pth -> .onnx
tools/prepare_onnx.py - фиксация dynamic axes в ONNX
tools/build_engine.py - компиляция .onnx -> .engine
models/               - данные, не хранятся в git
  pretrained/         - .pth файлы
  onnx/               - .onnx файлы
  engines/            - .engine файлы
videos/               - input/output видео, не хранятся в git
```

## Требования К Хосту

Для Docker-запусков с GPU нужны:

- NVIDIA driver на хосте
- Docker
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA runtime/CDI, настроенный для Docker

Настройка Docker после установки NVIDIA Container Toolkit:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка проброса GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Если Docker использует CDI, `nvidia-ctk cdi list` должен показывать устройства вроде
`nvidia.com/gpu=all`.

## Сборка Образа

```bash
DOCKER_BUILDKIT=1 docker build -t upscaler:latest .
```

Образ задаёт `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`. Это нужно для
NVDEC/NVENC через PyNvVideoCodec. Контейнеры запускаются с `--gpus all`.

Dockerfile отделяет тяжёлый dependency layer от application code:

* зависимости читаются из `pyproject.toml`/`uv.lock` через
  `uv sync --frozen --no-install-project` до копирования `upscaler/`, `tools/`
  и CLI-файлов;
* dependencies и финальная установка проекта выполняются через `uv sync`;
* `uv sync --inexact` используется намеренно, чтобы не удалять preinstalled
  NVIDIA/TensorRT packages из базового образа;
* повторная сборка после изменения Python-кода должна переиспользовать слой с
  `torch`, `cvcuda`, `pynvvideocodec`, `onnx`, `spandrel`;
* BuildKit cache mount используется для uv download/wheel cache и не попадает в
  production image layer.

Dev-образ с `ruff`/`mypy`:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_DEV=1 -t upscaler:dev .
```

## Docker Workflow

### 1. Экспорт `.pth` В ONNX

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  upscaler:latest export-onnx \
  --model_path models/pretrained/RealESRGAN_x2plus.pth
```

### 2. Подготовка ONNX

Используется, когда ONNX-модель имеет dynamic axes и для TensorRT нужны фиксированные
input shapes. Если `--size` не указан, создаются default variants для 1280x720 и
1920x1080.

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  upscaler:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720
```

### 3. Сборка TensorRT Engine

Компиляция обычно занимает 5-15 минут. Engine привязан к версии TensorRT и классу GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  upscaler:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/liveaction-span/engines/model_720p.engine \
  --timing-cache models/cache/trt.cache \
  --registry models/liveaction-span
```

`build-engine` автоматически создаёт sidecar manifest рядом с engine:

```text
models/liveaction-span/engines/model_720p.engine.json
```

В sidecar manifest сохраняются ONNX hash, engine hash, версия TensorRT, precision,
input/output shapes, profile и builder flags. Путь можно задать через `--manifest PATH`,
а отключить запись через `--no-manifest`.

Экспериментальный FP16 I/O engine можно собрать отдельным файлом:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  upscaler:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/liveaction-span/engines/model_720p_fp16io.engine \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry models/liveaction-span
```

`--fp16-io` меняет TensorRT input/output bindings на FP16. Это opt-in режим для
benchmark: проверяйте FPS, VRAM и визуальные артефакты отдельно от обычного FP32 I/O
engine. Для выбора FP16 I/O engine из registry используйте
`--engine-io-precision fp16`; для обычного engine — `--engine-io-precision fp32`.

Если передан `--registry models/liveaction-span`, команда также автоматически создаёт или
обновляет registry manifest:

```text
models/liveaction-span/manifest.json
```

Этот файл используется `--model models/liveaction-span`, чтобы выбрать подходящий static
engine по разрешению входного видео. Обычно оба manifest-файла генерируются командой
`build-engine`; вручную их писать не нужно.

Dynamic ONNX можно собрать напрямую, если явно задать TensorRT optimization profile:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  upscaler:latest build-engine \
  models/onnx/model.onnx \
  -o models/engines/model_dynamic_720p.engine \
  --min-shape input:1x3x360x640 \
  --opt-shape input:1x3x720x1280 \
  --max-shape input:1x3x1080x1920 \
  --timing-cache models/cache/trt.cache
```

Текущие команды video inference работают как static-shape full-frame path. Для
`upscale-video` и `upscale-video-nvcodec` используйте static ONNX variant из
`prepare-onnx` и собирайте static engine. Dynamic-profile build уже поддержан, но
dynamic runtime path остаётся будущей задачей.

### Model Registry

Model registry позволяет inference выбрать правильный static engine по разрешению
входного видео:

```text
models/liveaction-span/
  manifest.json
  engines/
    model_720p.engine
    model_720p.engine.json
    model_1080p.engine
    model_1080p.engine.json
```

Запуск через registry:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video \
  --model models/liveaction-span \
  --input videos/input.mp4
```

`--engine PATH` всё ещё поддерживается для явного выбора engine. Если в registry есть
несколько engines для одного разрешения, используйте `--engine-precision fp16|fp32`
и при необходимости `--engine-io-precision fp16|fp32` вместе с `--model`.

### 4. Апскейл Видео

Базовая команда: ffmpeg делает decode/encode, TensorRT inference выполняется на GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

NVDEC/NVENC backend: decode, color conversion, TensorRT inference и encode остаются на GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video-nvcodec \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

Выбор конкретного CUDA device:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest upscale-video-nvcodec \
  --gpu-id 1 \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

### 5. Benchmark

`benchmark` запускает один или несколько backend'ов на одинаковом input/engine и пишет
machine-readable JSON. Прогресс и diagnostics выводятся в `stderr`, поэтому `stdout`
можно безопасно использовать для JSON через `--json -`.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/artefacts:/app/artefacts" \
  upscaler:latest benchmark \
  --model models/liveaction-span \
  --input videos/input.mp4 \
  --backend ffmpeg,nvcodec \
  --warmup-frames 20 \
  --frames 300 \
  --json artefacts/benchmark.json
```

JSON содержит backend, engine, GPU, input/output resolution, `processed_frames`,
количество измеренных кадров, `processing_fps`, `throughput_fps`, `avg_frame_sec`,
`avg_frame_ms`, `min_frame_ms`, `max_frame_ms`, `cuda_graph_requested`,
`cuda_graph`, `cuda_graph_error`, `stage_ms` и `gpu_peak_mem_mb`.
Для benchmark команда фактически обрабатывает `warmup_frames + frames` кадров, а
первые warmup-кадры исключает из processing-метрик. `throughput_fps` считается по
полному wall-clock времени backend run.

Experimental CUDA Graph benchmark:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/artefacts:/app/artefacts" \
  upscaler:latest benchmark \
  --model models/liveaction-span \
  --input videos/input.mp4 \
  --backend nvcodec \
  --cuda-graph \
  --warmup-frames 20 \
  --frames 1000 \
  --json artefacts/benchmark_cuda_graph.json
```

`--cuda-graph` захватывает TensorRT enqueue для static-shape engine. Это opt-in
режим для benchmark; если CUDA Graph capture не поддержится конкретным runtime/stream,
pipeline откатится на обычный TensorRT enqueue.

Если JSON нужен напрямую в shell pipeline:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  upscaler:latest benchmark \
  --model models/liveaction-span \
  --input videos/input.mp4 \
  --backend nvcodec \
  --quiet \
  --json -
```

Важно про mount для JSON output: если указать `-v "$PWD:/app/artefacts"` и
`--json artefacts/result.json`, файл окажется в host `$PWD/result.json`, потому что
внутри контейнера `artefacts/result.json` резолвится как `/app/artefacts/result.json`.
Чтобы файл попал в host `./artefacts/result.json`, используйте
`-v "$PWD/artefacts:/app/artefacts"` или монтируйте весь репозиторий как `-v "$PWD:/app"`.

## CLI-Команды

Основные команды для видео:

```bash
upscale-video           # ffmpeg decode/encode + TensorRT
upscale-video-nvcodec   # NVDEC/NVENC + TensorRT
```

Алиасы для совместимости:

```bash
upscale      # alias for upscale-video
upscale-gpu  # alias for upscale-video-nvcodec
```

Команды для подготовки моделей:

```bash
export-onnx
prepare-onnx
build-engine
benchmark
benchmark-video
```

Общие inference options:

```bash
--engine PATH       путь к .engine файлу
--model PATH        директория model registry или manifest JSON
--input PATH        входное видео
--output PATH       выходное видео, default: *_upscaled.ext
--gpu-id N          CUDA GPU index, default: 0
--engine-precision  предпочесть fp16 или fp32 при использовании --model
--engine-io-precision  предпочесть fp16 или fp32 I/O при использовании --model
--max-frames N      ограничить количество кадров, 0 = все
--profile           вывести per-stage profiling
--verbose           подробный вывод
--quiet             минимальный вывод
```

Backend-specific options:

```bash
upscale-video:         --crf N
upscale-video-nvcodec: --crf N --codec h264|hevc
```

Важно: `--crf` в NVDEC/NVENC backend мапится в оценочный NVENC bitrate. Это не то же
самое, что x264 CRF.

## Docker Compose

ffmpeg backend:

```bash
docker compose run --rm upscale-video
```

NVDEC/NVENC backend:

```bash
docker compose run --rm upscale-video-nvcodec
```

Пути к engine/model и input video задаются в `docker-compose.yml`.

## Установка Для Разработки

Локальная установка из исходников нужна только для разработки. Для обычных запусков
поддерживаемый workflow — Docker.

```bash
uv sync --extra ffmpeg --group dev
uv sync --extra gpu --group dev
uv sync --extra export --group dev
```

## Проверки Качества

Локальные проверки:

```bash
ruff check .
mypy .
python3 -m compileall -q benchmark.py inference.py inference_gpu.py tools upscaler
```

Docker-based проверки:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_DEV=1 -t upscaler:dev .
docker run --rm -v "$PWD:/app" upscaler:dev ruff check .
docker run --rm -v "$PWD:/app" upscaler:dev mypy .
docker run --rm -v "$PWD:/app" upscaler:dev \
  python3 -m compileall -q benchmark.py inference.py inference_gpu.py tools upscaler
```
