# AI Media Enhancer - актуальный контекст проекта

## Текущее состояние

Репозиторий: `ai-media-enhancer`

CLI-инструменты для AI-обработки медиа через TensorRT. Текущий реализованный
workflow - апскейл видео; поддерживаются модели RealESRGAN и SPAN в форматах
`.pth` и ONNX.

Разработка ведется локально. Запуски с тяжелыми зависимостями выполняются в Docker,
обычно на удаленном сервере с GPU.

Основной рабочий сценарий - Docker-first. Локальная установка из исходников нужна
только для разработки.

Production runtime сейчас привязан к Python 3.12 из базового TensorRT Docker image
`nvcr.io/nvidia/tensorrt:26.06-py3`.

## Правила работы агента

- Основной workflow - Docker-first. Локально часто нет TensorRT/PyNvVideoCodec/CV-CUDA,
  поэтому GPU/runtime проверки выполняются в контейнере на GPU-хосте.
- Не коммитить `models/`, `videos/` и большие runtime artefacts без явной команды.
- Источники моделей фиксировать в `docs/MODELS.md` и `model_sources.json`; веса,
  ONNX и TensorRT engines не vendored в репозиторий.
- Перед полным batch-прогоном сначала делать короткий smoke через `--max-frames`.
- Для изменений в color/encoding path проверять не только запуск, но и `ffprobe`
  output: `pix_fmt`, `color_range`, `color_space`, `color_transfer`,
  `color_primaries`, bitrate, duration и frame count.
- Заметные изменения workflow, CLI, Docker, структуры файлов и проектных правил
  фиксировать в `docs/CHANGES.md`.
- Performance-изменения фиксировать в `docs/PERFORMANCE_LOG.md` только вместе с
  измерением: что изменилось, какой benchmark/команда, какой прирост или регресс.
- Если проверку нельзя выполнить локально из-за отсутствующих GPU/runtime зависимостей,
  явно указать это в финальном ответе.
- Тесты запускать через Docker dev image. Unit tests должны оставаться pure-Python и
  не импортировать TensorRT, CV-CUDA или PyNvVideoCodec.

## Структура файлов

```
.
├── AGENTS.md
├── README.md
├── docs/
│   ├── ROADMAP.md              # короткий актуальный план
│   ├── CHANGES.md              # журнал заметных проектных изменений
│   ├── MODELS.md               # источники моделей, лицензии и локальные пути
│   ├── TESTING.md              # архитектура тестов и Docker-only workflow
│   └── PERFORMANCE_LOG.md      # журнал performance-изменений и benchmark results
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── model_sources.json           # машинно-читаемый каталог upstream model sources
├── ai_media/
│   ├── cli/                     # console entrypoints
│   ├── pipelines/               # ffmpeg и NVDEC/NVENC backends
│   ├── runtime/                 # TensorRT runtime и RuntimeEngine protocol
│   ├── video/                   # ffprobe metadata и colorspace helpers
│   ├── models/                  # ModelSpec и engine registry
│   └── profiling.py
└── models/                      # данные, игнорируются git
    ├── pretrained/
    ├── onnx/
    └── engines/
```

## Требования для Docker с GPU

Для Docker-запусков, которым нужен GPU, на хосте должны быть:

- NVIDIA driver
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

Если Docker использует CDI, команда `nvidia-ctk cdi list` должна показывать устройства
вроде `nvidia.com/gpu=all`.

Docker-образ проекта задает:

```dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

Это нужно для NVENC/NVDEC через PyNvVideoCodec.

## CLI-команды

Основная команда для видео:

```bash
upscale --backend ffmpeg    # ffmpeg decode/encode + TensorRT
upscale --backend nvcodec   # NVDEC/NVENC + cvcuda + TensorRT
```

Команды для подготовки моделей и benchmark:

```bash
export-onnx
prepare-onnx
build-engine
benchmark-upscale
```

## Docker workflow

Сборка образа:

```bash
DOCKER_BUILDKIT=1 docker build -t ai-media-enhancer:latest .
```

Dockerfile устроен так, чтобы code-only изменения не инвалидировали тяжёлый
dependency layer:

- dependencies читаются из `pyproject.toml`/`uv.lock` через
  `uv export --frozen --no-emit-project` до копирования application code;
- dependencies ставятся в venv `/opt/ai-media-enhancer` с `--system-site-packages`, чтобы видеть
  preinstalled packages из TensorRT base image и не менять managed `/usr`;
- `RUN --mount=type=cache,target=/root/.cache/uv` использует BuildKit uv cache;
- сам проект устанавливается после копирования кода через `uv pip install --python "$VIRTUAL_ENV" --no-deps .`.

Dev-образ с `ruff`/`mypy`:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_DEV=1 -t ai-media-enhancer:dev .
```

Экспорт `.pth` в ONNX. GPU не нужен:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest export-onnx \
  --model_path models/pretrained/RealESRGAN_x2plus.pth
```

Подготовка dynamic ONNX в static variants. GPU не нужен:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest prepare-onnx \
  models/onnx/model.onnx \
  --size 1280x720
```

Если `--size` не указан, `prepare-onnx` создаёт default variants для 1280x720 и
1920x1080.

Сборка TensorRT engine. GPU нужен:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/liveaction-span/engines/model_720p.engine \
  --timing-cache models/cache/trt.cache \
  --registry models/liveaction-span
```

`build-engine` по умолчанию пишет sidecar manifest в `<engine>.json`. Там фиксируются
ONNX hash, engine hash, TensorRT version, precision, input/output shapes, profile и
builder flags. Путь можно изменить через `--manifest PATH`, отключить через
`--no-manifest`. `--registry models/liveaction-span` дополнительно обновляет
`models/liveaction-span/manifest.json`.

Experimental FP16 I/O engine:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/onnx/model_720p.onnx \
  -o models/liveaction-span/engines/model_720p_fp16io.engine \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry models/liveaction-span
```

`--fp16-io` меняет TensorRT input/output bindings на FP16. Это opt-in benchmark path;
обычный FP16 engine продолжает использовать FP32 I/O. Для выбора из registry
использовать `--engine-io-precision fp16|fp32` вместе с `--model`.

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

Запуск ffmpeg backend:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend ffmpeg \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

Запуск через model/engine registry:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend nvcodec \
  --model models/liveaction-span \
  --input videos/input.mp4
```

Запуск NVDEC/NVENC backend:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/videos:/app/videos" \
  ai-media-enhancer:latest upscale \
  --backend nvcodec \
  --engine models/engines/model_720p.engine \
  --input videos/input.mp4
```

## Проверки качества

Проверочные инструменты устанавливаются в Docker dev image.

Правила работы с проверками:

- После любых Python-изменений минимум запускать `ruff check .` в Docker dev image.
- Перед коммитом с изменениями Python-кода запускать полный Docker gate: `ruff`,
  `mypy`, `compileall`, `pytest -q tests/unit`.
- `ruff check . --fix` допустим только для механических исправлений; после него нужно
  посмотреть diff и не принимать автоправки вслепую.
- Не передавать Markdown-файлы в `ruff` явно. `ruff check .` сам применяет конфиг
  проекта и проверяет Python scope.
- Если локально нет runtime-only зависимостей, использовать Docker-based вариант ниже.

Docker-based вариант:

```bash
make build-dev
make check
```

`mypy` сейчас настроен как инкрементальный gate: проверяет весь текущий код с
`check_untyped_defs`, игнорирует отсутствующие runtime-only зависимости и временно
подавляет `union-attr` до выделения явного runtime interface в Stage 1B.

Unit tests описаны в `docs/TESTING.md`. Они не должны импортировать runtime-only
модули и не должны требовать `--gpus all`.

## Как устроен инференс

В проекте есть два video inference backend. Оба используют TensorRT на GPU, но
отличаются тем, где выполняются decode, color conversion и encode.

### Общая часть

Общий жизненный цикл задает `BasePipeline` в `ai_media/pipelines/base.py`:

1. Получает уже распарсенные CLI-аргументы из `ai_media/cli/upscale.py`.
2. Проверяет наличие `--engine` или `--model`, а также `--input`.
3. Через `ffprobe` читает параметры видео в `VideoInfo`: ширина, высота, FPS,
   количество кадров и доступные color metadata.
4. Если задан `--model`, выбирает static engine из model registry по разрешению видео;
   если задан `--engine`, использует его напрямую.
5. Создает `TensorRTRuntime` из выбранного `.engine` через общий `RuntimeEngine` interface.
6. Валидирует минимальный `ModelSpec`: static single-frame RGB upscale, NCHW,
   batch=1, fp32, range `0_1`, равномерный integer scale.
7. Проверяет, что размер входного видео совпадает с input shape engine.
8. Инициализирует decoder и encoder выбранного backend.
9. Запускает цикл обработки кадров.
10. Печатает статистику и, если включен `--profile`, таблицу профилирования.

`TensorRTRuntime` в `ai_media/runtime/tensorrt.py`:

- загружает serialized TensorRT engine;
- создает execution context;
- читает input/output tensor names и shapes;
- строит и валидирует `ModelSpec` до выделения GPU buffers;
- заранее выделяет `gpu_input` и `gpu_output` на выбранном `cuda:<gpu-id>`;
- привязывает GPU buffers к TensorRT context через `set_tensor_address`;
- выполняет inference через `execute_async_v3`;
- скрывает `context`, `gpu_input`, `gpu_output` от video pipeline;
- позволяет caller передать CUDA stream; если stream не передан, runtime использует
  свой stream и сам синхронизируется.

Общий флаг `--gpu-id` выбирает CUDA GPU для TensorRT. В NVDEC/NVENC backend этот же
ID используется для PyNvVideoCodec decode/encode.

### `upscale --backend ffmpeg`

Файл: `ai_media/pipelines/ffmpeg.py`

Путь данных:

```text
ffmpeg decode (CPU) -> RGB raw pipe -> numpy -> torch cuda -> TensorRT
-> torch output -> CPU numpy -> RGB raw pipe -> ffmpeg encode (CPU)
```

Подробно:

1. `setup_decoder()` запускает `ffmpeg` как subprocess.
2. ffmpeg декодирует входное видео и пишет `rgb24` rawvideo в `stdout`.
3. Python читает из pipe ровно один RGB frame размером `input_w * input_h * 3`.
4. Frame превращается в `numpy.ndarray` формы `[H, W, 3]`.
5. `TRTInference.infer()` переносит кадр в CUDA, делает `permute` в `[1, 3, H, W]`,
   приводит к `float32` и нормализует в диапазон `0..1`.
6. TensorRT выполняет inference.
7. Output TensorRT приводится обратно к `uint8 RGB`.
8. Результат копируется на CPU как numpy.
9. Python пишет raw RGB frame в `stdin` encoder subprocess.
10. ffmpeg кодирует выходное видео через `libx264`, сохраняет аудио через `-c:a copy`.

Этот backend проще и надежнее по зависимостям, но использует CPU pipe и CPU encode/decode.
Он все равно GPU backend в части нейросетевого inference, потому что TensorRT работает
на CUDA.

Качество видео задается настоящим x264 `--crf`.

### `upscale --backend nvcodec`

Файл: `ai_media/pipelines/nvcodec.py`

Путь данных:

```text
NVDEC (GPU) -> NV12 GPU surface -> cvcuda RGB -> TensorRT
-> cvcuda NV12 -> NVENC (GPU) -> raw H.264/HEVC -> ffmpeg mux
```

Подробно:

1. `setup_decoder()` создает `PyNvVideoCodec.ThreadedDecoder`.
2. Decoder читает compressed video и декодирует кадры через NVDEC.
3. Кадры выдаются как NV12 GPU surfaces в device memory.
4. Python получает frame и делает `torch.from_dlpack(raw_frame)`, то есть берет GPU
   данные без CPU copy.
5. Per-job `FrameBufferPool` переиспользует GPU buffers для NV12/RGB/NCHW hot path.
6. `nv12_to_rgb_into()` через cvcuda конвертирует NV12 в RGB на GPU с явным
   SDR color spec (`BT709` для HD/UHD, `BT601` для SD).
7. RGB tensor приводится к формату TensorRT input: `[1, 3, H, W]`, `float32`, `0..1`.
   Для NVDEC/NVENC backend это делается через `infer_rgb_tensor_into()` с
   preallocated buffers.
8. TensorRT выполняет inference.
9. Output приводится к `uint8 RGB` на GPU.
10. `rgb_to_nv12_into()` через cvcuda конвертирует RGB обратно в NV12 на GPU
    с тем же color spec.
11. NV12 tensor передается в NVENC через PyNvVideoCodec.
12. NVENC пишет raw H.264 или HEVC bitstream во временный файл.
13. В `finalize()` вызывается `ffmpeg`, который mux-ит raw video stream с аудио из
    исходного файла в финальный MP4.

В этом backend основная data path остается на GPU. CPU участвует в orchestration,
записи raw bitstream и финальном mux, но не гоняет кадры туда-сюда как numpy buffers.

Качество задается автоматически или явным `--bitrate-mbps`. Если `--bitrate-mbps`
не указан, NVENC backend оценивает target bitrate от source video bitrate:
`source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`. Если `ffprobe` не смог
определить bitrate исходного видео, NVENC backend требует явный `--bitrate-mbps`
и завершится с ошибкой. `--crf` поддерживается только для ffmpeg backend.

### Профилирование

Флаг `--profile` включает `ProfileCollector`.

В ffmpeg backend профиль включает:

- `ffmpeg decode (pipe read)` как wall-clock;
- `Preprocess (CPU->GPU)` через CUDA events;
- `TRT inference` через CUDA events;
- `Postprocess (GPU->CPU)` через CUDA events;
- `ffmpeg encode (pipe write)` как wall-clock.

В NVDEC/NVENC backend профиль включает:

- `NV12->RGB (cvcuda)`;
- `TRT inference`;
- `RGB->NV12 (cvcuda)`;
- `NVENC encode`.

Текущий profiler измеряет processing stages после получения кадра из decoder. Он не
является полным end-to-end профилем всего процесса для всех backend.

`--profile-json PATH` пишет machine-readable summary для одного запуска backend.
Команда `benchmark-upscale` запускает `upscale --backend ffmpeg` и/или
`upscale --backend nvcodec` на одинаковом input/engine и собирает итоговый JSON:

```bash
benchmark-upscale \
  --model models/liveaction-span \
  --input videos/input.mp4 \
  --backend ffmpeg,nvcodec \
  --warmup-frames 20 \
  --frames 300 \
  --json artefacts/benchmark.json
```

Benchmark обрабатывает `warmup_frames + frames` кадров, исключает warmup из метрик и
пишет `processing_fps`, `throughput_fps`, `avg_frame_sec`, `avg_frame_ms`,
`min_frame_ms`, `max_frame_ms`, `cuda_graph_requested`, `cuda_graph`,
`cuda_graph_error`, stage timings, GPU name, peak GPU memory и выбранный engine.
`processing_fps` исключает warmup, `throughput_fps` считается по полному wall-clock
времени backend run.
`--cuda-graph` включает experimental CUDA Graph capture для TensorRT enqueue в
static-shape runtime. Если capture не поддерживается конкретным stream/runtime,
pipeline пишет warning и откатывается на обычный TensorRT enqueue.
`--json -` пишет чистый JSON в stdout; progress и diagnostics benchmark идут в
stderr. `--quiet` подавляет служебные progress-строки benchmark, ошибки всё равно
пишутся в stderr.

## Важные заметки про TensorRT

TensorRT engine привязан к версии TensorRT и классу GPU. При переносе между существенно
разными GPU или версиями TensorRT container engine лучше пересобрать.

Dynamic ONNX нельзя напрямую собрать через `build-engine`, если не задан TensorRT
optimization profile. Есть два поддержанных workflow:

1. Запустить `prepare-onnx` на dynamic ONNX и собрать сгенерированный static файл.
2. Передать `--min-shape`, `--opt-shape`, `--max-shape` в `build-engine`.

Если TensorRT печатает input/output shape вида `(-1, 3, -1, -1)`, ONNX все еще dynamic,
и его можно передавать в `build-engine` только с explicit optimization profile.

Текущий video inference runtime работает как static-shape full-frame path. Dynamic ONNX
с TensorRT profile можно собрать, но запуск такого engine в `upscale --backend ...`
пока не является поддержанным runtime path без отдельной
логики выбора concrete shape и перевыделения buffers.

## Заметки по производительности

Предыдущие измерения на L4 с SPAN 720p -> 1440p:

- TensorRT inference доминировал по времени кадра, около 98% в профилированном запуске.
- ffmpeg pipe backend и NVDEC/NVENC backend давали похожий FPS на тяжелых моделях.
- NVDEC/NVENC backend в первую очередь снижает нагрузку на CPU/RAM и должен быть
  полезнее на более легких и быстрых моделях.

## Текущие известные ограничения

- Auto bitrate в NVENC backend является эвристикой от source bitrate. Для
  воспроизводимого размера output использовать явный `--bitrate-mbps`.
- Runtime рассчитан на SDR 8-bit video. HDR/P010/yuv422/yuv444 требуют отдельной
  цветовой политики/tonemap и сейчас должны отклоняться fail-fast.
