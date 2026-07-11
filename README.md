# AI Media Enhancer

CLI-инструменты для AI-обработки медиа через TensorRT. Текущий реализованный
workflow - апскейл видео; поддерживаются модели RealESRGAN и SPAN в форматах
`.pth` и ONNX.

Рекомендуемый workflow — Docker. Образ содержит runtime-зависимости для TensorRT
inference, NVDEC/NVENC inference, подготовки ONNX и экспорта моделей.

Production runtime сейчас привязан к Python 3.12 из базового TensorRT Docker image
`nvcr.io/nvidia/tensorrt:26.06-py3`.

Проверялось на Tesla T4.

## Структура

```text
ai_media/cli/        - CLI entrypoints
ai_media/pipelines/  - ffmpeg и NVDEC/NVENC pipeline backends
ai_media/runtime/    - TensorRT runtime wrapper и runtime protocol
ai_media/video/      - video metadata и GPU colorspace helpers
ai_media/models/     - model contracts для runtime validation
docs/ROADMAP.md      - короткий актуальный план
docs/CHANGES.md      - журнал заметных проектных изменений
docs/MODELS.md       - источники моделей, лицензии и локальные пути
docs/TESTING.md      - архитектура тестов и Docker-only test workflow
docs/PERFORMANCE_LOG.md - журнал performance-изменений
model_sources.json   - машинно-читаемый каталог upstream model sources
models/               - данные, не хранятся в git
  pretrained/         - .pth файлы
  onnx/               - .onnx файлы
  engines/            - .engine файлы
videos/               - input/output видео, не хранятся в git
```

## Требования К Хосту

Для GPU-запусков нужен хост, на котором уже настроены:

- NVIDIA driver;
- Docker;
- GPU passthrough в Docker для `docker run --gpus all`.

## Сборка Образа

```bash
DOCKER_BUILDKIT=1 docker build -t ai-media-enhancer:latest .
```

Образ задаёт `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`. Это нужно для
NVDEC/NVENC через PyNvVideoCodec. Контейнеры запускаются с `--gpus all`.

Dockerfile отделяет тяжёлый dependency layer от application code:

* зависимости читаются из `pyproject.toml`/`uv.lock` через
  `uv export --frozen --no-emit-project` до копирования `ai_media/`;
* dependencies ставятся в venv `/opt/ai-media-enhancer` с `--system-site-packages`, чтобы видеть
  preinstalled NVIDIA/TensorRT packages из базового образа и не менять managed `/usr`;
* повторная сборка после изменения Python-кода должна переиспользовать слой с
  `torch`, `cvcuda`, `pynvvideocodec`, `onnx`, `onnxscript`, `spandrel`;
* BuildKit cache mount используется для uv download/wheel cache и не попадает в
  production image layer.

Dev-образ с `ruff`/`mypy`:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_DEV=1 -t ai-media-enhancer:dev .
```

## Модели

Веса моделей, ONNX-файлы и TensorRT engines не входят в репозиторий. Источники и
лицензии RealESRGAN/SPAN описаны в `model_sources.json`, локальные пути и workflow -
в `docs/MODELS.md`.

## Docker Workflow

### 1. Экспорт `.pth` В ONNX

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest export-onnx \
  --model_path models/pretrained/RealESRGAN_x2plus.pth
```

### 2. Подготовка ONNX

Используется, когда ONNX-модель имеет dynamic axes и для TensorRT нужны фиксированные
input shapes. Если `--size` не указан, создаются default variants для 1280x720 и
1920x1080.

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720
```

Для TensorRT 11 FP16 задаётся не builder flag, а типами внутри ONNX. Mixed-precision
variant создаётся на этом этапе:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720 \
  --precision fp16
```

`--precision fp16` делает лёгкий ONNX graph rewrite через `onnxconverter-common`:
внутренние float tensors переводятся в FP16, а input/output тензоры остаются FP32,
чтобы не менять текущий video runtime contract. GPU для этого шага не нужен.

### 3. Сборка TensorRT Engine

Компиляция обычно занимает 5-15 минут. Engine привязан к версии TensorRT и классу GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/liveaction-span/engines/model_720p.engine \
  --timing-cache models/cache/trt.cache
```

`build-engine` автоматически создаёт sidecar manifest рядом с engine:

```text
models/liveaction-span/engines/model_720p.engine.json
```

В sidecar manifest сохраняются ONNX hash, engine hash, версия TensorRT, precision,
input/output shapes, profile и builder flags. Путь можно задать через `--manifest PATH`,
а отключить запись через `--no-manifest`.

FP16 engine собирается из FP16/mixed-precision ONNX без дополнительных precision-флагов
в `build-engine`:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p_fp16.onnx \
  -o models/liveaction-span/engines/model_720p_fp16.engine \
  --timing-cache models/cache/trt.cache
```

В TensorRT 11 weak-typing флаги вроде `BuilderFlag.FP16` удалены. Если нужен FP16,
сначала создайте ONNX через `prepare-onnx --precision fp16`, затем передайте этот ONNX
в `build-engine`.

Dynamic ONNX можно собрать напрямую, если явно задать TensorRT optimization profile:

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

Текущая команда video upscale работает как static-shape full-frame path. Для
`upscale --backend ffmpeg|nvcodec` используйте static ONNX variant из
`prepare-onnx` и собирайте static engine. Dynamic-profile build уже поддержан, но
dynamic runtime path остаётся будущей задачей.

### 4. Апскейл Видео

`--engine` обязателен и должен соответствовать разрешению входного видео. Runtime
проверяет input shape engine и завершает запуск при несовпадении.

ffmpeg backend: ffmpeg делает decode/encode, TensorRT inference выполняется на GPU.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend ffmpeg \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

NVDEC/NVENC backend: decode, color conversion, TensorRT inference и encode остаются на GPU.

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

Выбор конкретного CUDA device:

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

### 5. Benchmark

`benchmark-upscale` запускает один или несколько backend'ов на одинаковом input/engine и пишет
machine-readable JSON. Прогресс и diagnostics выводятся в `stderr`, поэтому `stdout`
можно безопасно использовать для JSON через `--json -`.

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/artefacts:/app/artefacts" \
  ai-media-enhancer:latest benchmark-upscale \
  --engine models/liveaction-span/engines/model_720p.engine \
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
Для benchmark-upscale команда фактически обрабатывает `warmup_frames + frames` кадров, а
первые warmup-кадры исключает из processing-метрик. `throughput_fps` считается по
полному wall-clock времени backend run.

Experimental CUDA Graph benchmark:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  -v "$PWD/artefacts:/app/artefacts" \
  ai-media-enhancer:latest benchmark-upscale \
  --engine models/liveaction-span/engines/model_720p.engine \
  --input videos/input.mp4 \
  --backend nvcodec \
  --cuda-graph \
  --warmup-frames 20 \
  --frames 1000 \
  --json artefacts/benchmark_cuda_graph.json
```

`--cuda-graph` захватывает TensorRT enqueue для static-shape engine. Это opt-in
режим для benchmark-upscale; если CUDA Graph capture не поддержится конкретным runtime/stream,
pipeline откатится на обычный TensorRT enqueue.

Если JSON нужен напрямую в shell pipeline:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest benchmark-upscale \
  --engine models/liveaction-span/engines/model_720p.engine \
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

Основная команда для видео:

```bash
upscale --backend ffmpeg    # ffmpeg decode/encode + TensorRT
upscale --backend nvcodec   # NVDEC/NVENC + TensorRT
```

Команды для подготовки моделей и benchmark:

```bash
export-onnx
prepare-onnx
build-engine
benchmark-upscale
```

Общие inference options:

```bash
--engine PATH       путь к .engine файлу
--backend BACKEND   ffmpeg или nvcodec, default: nvcodec
--input PATH        входное видео
--output PATH       выходное видео, default: *_upscaled.ext
--gpu-id N          CUDA GPU index, default: 0
--max-frames N      ограничить количество кадров, 0 = все
--profile           вывести per-stage profiling
--verbose           подробный вывод
--quiet             минимальный вывод
```

Backend options:

```bash
--crf N          ffmpeg/libx264 CRF, default: 18; не поддерживается nvcodec
--bitrate-mbps N  явный target bitrate для nvcodec backend, перекрывает auto
--codec h264|hevc   используется nvcodec backend, ffmpeg backend пока игнорирует
```

Важно: если `--bitrate-mbps` не указан, NVDEC/NVENC backend автоматически оценивает
target bitrate от source video bitrate:
`source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`. Это снижает риск случайно
получить огромный output после апскейла. Для полностью контролируемого размера
используйте явный `--bitrate-mbps`.

Если `ffprobe` не смог определить bitrate исходного видео, NVDEC/NVENC backend
требует явный `--bitrate-mbps` и завершится с ошибкой. `--crf` поддерживается
только для ffmpeg backend.

Color contract: текущий runtime рассчитан на SDR 8-bit video. `nvcodec` backend
fail-fast отклоняет не-`yuv420p`/`nv12` inputs и HDR transfer functions; для
NV12/RGB conversion используется явный CV-CUDA color spec, а output получает
явные color tags вместо `unknown`.

## Docker Compose

ffmpeg backend:

```bash
docker compose run --rm upscale-ffmpeg
```

NVDEC/NVENC backend:

```bash
docker compose run --rm upscale-nvcodec
```

Пути к engine и input video задаются в `docker-compose.yml`.

## Установка Для Разработки

Локальная установка из исходников нужна только для разработки. Для обычных запусков
поддерживаемый workflow — Docker.

```bash
uv sync --extra ffmpeg --group dev
uv sync --extra gpu --group dev
uv sync --extra export --group dev
```

## Проверки Качества

Проверки запускаются через Docker dev image. Unit tests не требуют GPU и не должны
импортировать TensorRT, CV-CUDA или PyNvVideoCodec.

Docker-based проверки:

```bash
make build-dev
make check
```

Подробная архитектура тестовых слоёв описана в `docs/TESTING.md`.
