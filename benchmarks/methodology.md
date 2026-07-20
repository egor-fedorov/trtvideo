# Benchmark Methodology

## Назначение

Benchmark проверяет тезис, что GPU-resident pipeline
`NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` обеспечивает высокий
end-to-end throughput без выгрузки несжатых кадров на CPU.

Результаты делятся на три класса и не смешиваются:

1. `trtexec` показывает inference ceiling того же TensorRT engine без video I/O.
2. `vs-mlrt/vstrt` сравнивает TensorRT integration при одинаковой модели и
   максимально близком video/encoding contract.
3. Video2X является product-level comparison, если одинаковая модель или
   encoder contract недоступны.

`ffmpeg` backend проекта используется только как diagnostic baseline. Основной
публичный результат `ai-media-enhancer` снимается с `--backend nvcodec`.

## Workload

Основная модель - `RealESRGAN_x2plus` из официального release `v0.2.1`.
Для `ai-media-enhancer`, `vstrt` и `trtexec` используется один исходный ONNX
tensor contract: batch 1, RGB NCHW, FP32 input/output, mixed-FP16 internal graph,
static full-frame shape.

Основной input создаётся из lossless Sintel trailer 1080p24 Y4M. Исходник и
подготовленные assets не коммитятся; URL, SHA256 и attribution зафиксированы в
`workloads/realesrgan_x2plus_sintel.json`.

Подготавливаются два video-only H.264 input:

| Workload | Input | Output | Frames | FPS |
|---|---:|---:|---:|---:|
| primary | 1920x1080 | 3840x2160 | 1000 | 24/1 |
| secondary | 1280x720 | 2560x1440 | 1000 | 24/1 |

Оба input имеют `yuv420p`, limited-range BT.709, SAR 1:1 и не содержат audio,
subtitles, chapters или пользовательскую metadata.

## Output Contract

Для полного межпродуктового сравнения используется одинаковый NVENC contract:

- H.264/NV12;
- preset P4, high-quality tuning;
- 60 Mbps для 4K и 35 Mbps для 1440p;
- B-frames отключены;
- GOP и IDR interval около одной секунды;
- FPS input сохраняется;
- limited-range BT.709 tags задаются явно;
- audio, subtitles и chapters отсутствуют.

Перед timed suite выполняется untimed output smoke. Если конкурент не позволяет
воспроизвести contract или фактический video bitrate отличается от target либо
других сравниваемых outputs более чем на 10%, результат помечается как
product-level и не используется для прямого pipeline claim.

Output считается валидным только после полного decode через `ffmpeg -f null -` и
проверки resolution, codec, pixel format, color tags, FPS, duration, frame count,
B-frames, keyframe interval и монотонности PTS/DTS. Автоматизация этой проверки
относится к Stage 1.

## Timing Contract

Per-stage profiling отключается: не использовать `--profile`, `--profile-json`
или текущий `benchmark-upscale`, пока Stage 1 не отделит machine-readable
end-to-end metrics от profiling instrumentation.

Для каждого measured run:

1. Отдельный discarded процесс обрабатывает первые 100 кадров.
2. Сразу после него новый процесс обрабатывает ровно 1000 кадров.
3. Внешний monotonic timer запускается перед созданием процесса и останавливается
   после успешного process exit, когда encode, flush и mux завершены.
4. stdout/stderr не выводятся в интерактивный terminal, кроме сохранённого лога.
5. Input и output находятся на одном локальном filesystem для всех продуктов.

Основные метрики - полный wall time и `1000 / wall_time` end-to-end FPS.
Inference-only и stage timings публикуются отдельно и не заменяют end-to-end FPS.

Для каждой комбинации выполняются минимум три measured run. Порядок продуктов
ротируется между раундами. Публикуются все raw values, median, min/max и relative
spread `(max - min) / median`. При spread больше 5% выполняются ещё два run. Если
пять run остаются нестабильными, результат маркируется как unstable.

CUDA Graph не включается в основной baseline, пока это experimental opt-in
режим проекта. На Stage 2 graph enabled/disabled сравниваются попарно с одинаковым
режимом `trtexec`.

## Environment Contract

До benchmark необходимо выбрать одну физическую NVIDIA GPU. Все engines и все
сравниваемые результаты строятся и снимаются на этой карте. Результаты разных GPU
не объединяются в одну таблицу.

Benchmark host должен:

- не обслуживать display workload на benchmark GPU;
- не иметь других compute/video процессов во время run;
- использовать неизменные driver, power limit и clock policy;
- выполнять одинаковый 100-frame warmup перед каждым measured run;
- выдерживать одинаковый idle interval между run;
- ротировать порядок продуктов;
- отклонять run при thermal/power throttling.

Публичный environment report строится только по allowlist:

- GPU model, compute capability и total VRAM;
- CPU model и logical core count;
- driver, CUDA, TensorRT, CV-CUDA, PyNvVideoCodec, FFmpeg и Python versions;
- Docker image references и immutable digests;
- commit/version конкурентов;
- power limit, clock policy, temperature и throttle state;
- repository commit;
- SHA256 input, weights, ONNX, engine и sidecar manifest;
- точные команды и benchmark parameters.

Не сохраняются hostname, username, IP, GPU UUID/serial, container IDs,
абсолютные host paths и произвольный environment dump.

## Run Validity

Measured run недействителен, если:

- asset checksum или engine/model contract не совпадает;
- output validation завершилась ошибкой;
- обработано не ровно 1000 кадров;
- изменился environment, image, command или engine между сравниваемыми run;
- обнаружены посторонняя GPU-нагрузка или throttling;
- включён per-frame profiler;
- фактические output settings не соответствуют заявленному классу сравнения.

Stage 0 считается закрытым после успешного `make benchmark-verify` на любом
хосте. Выбор одной физической benchmark GPU и фиксация её environment являются
отдельным prerequisite Stage 2 перед снятием performance baseline.
