# Benchmark Methodology

## Назначение

Benchmark проверяет, сохраняет ли GPU-resident pipeline
`NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC` преимущество на полном пути от
compressed input до валидного MP4 output.

Результаты не смешиваются:

1. `vstrt parity` - проект и локально собранный VapourSynth/vstrt используют один
   TensorRT 11 engine. Это сравнение integration и video pipeline.
2. `VSGAN product` - проект сравнивается с pinned stock
   VSGAN-tensorrt-docker. Используется один ONNX, но отдельные native engines:
   stock VSGAN работает на TensorRT 10.16, проект - на TensorRT 11.
3. `trtexec diagnostic` - inference ceiling без decode, colorspace, encode и mux.

Video2X не входит в матрицу: доступная версия не поддерживает canonical
`RealESRGAN_x2plus` и выполняет другую anime-модель. Его FPS нельзя использовать
для same-model performance claim.

Основной backend проекта - `nvcodec`. `ffmpeg` backend остаётся diagnostic
baseline и не заменяет GPU-resident результат.

## Workloads

Обязательны две x2-модели:

- `RealESRGAN_x2plus` - тяжёлый model-bound workload;
- `2xLiveActionV1_SPAN` - лёгкий workload для измерения pipeline overhead.

Обе модели экспортируются через Spandrel в static full-frame ONNX. Canonical
tensor contract: batch 1, RGB NCHW, FP32 input/output bindings, mixed-FP16 graph,
tiling disabled.

Основной input создаётся из lossless Sintel trailer 1080p24 Y4M. Подготавливаются
два video-only H.264 input:

| Режим | Input | Output | Frames | FPS |
|---|---:|---:|---:|---:|
| primary | 1920x1080 | 3840x2160 | 1000 | 24/1 |
| confirmation | 1280x720 | 2560x1440 | 1000 | 24/1 |

Input использует `yuv420p`, limited BT.709, SAR 1:1, B-frames 0 и GOP 24. Audio,
subtitles, chapters и пользовательская metadata отсутствуют. URLs, hashes,
licenses и attribution находятся в `workloads/`.

После canonical campaign headline workload повторяется на коротком live-action
clip с большим движением. Его результаты публикуются отдельно от Sintel.

## Inference Contract

Technical parity требует:

```text
engine SHA256 identical
input/output dtype identical
static input/output shape identical
batch size = 1
full-frame processing
tiling disabled
requests = 1
TensorRT streams = 1
CUDA Graph disabled
```

Stock VSGAN не может загрузить TRT11 engine. Для product comparison оба engine
строятся из одного canonical ONNX на одной GPU с одинаковыми shape, dtype, batch
и builder intent. Engine hash и TensorRT runtime будут различаться и должны быть
показаны явно. VSGAN фиксируется immutable image digest и source revision;
разрешены только `.vpy` configuration, model/engine mount и encoder adapter.
Stock inference stack не изменяется. Внешний `vspipe | ffmpeg` encode нормализован
к pinned Ubuntu FFmpeg `7:6.1.1-3ubuntu5`: upstream binary требует NVENC API 13.1
и driver 610+, отсутствующий на benchmark host. Эта адаптация показывается в
implementation metadata; изменение внутреннего кода VSGAN считается fork.

CUDA Graph не включается в parity baseline. Текущая реализация проекта захватывает
только TensorRT call и остаётся experimental. Graph-enabled режим исследуется
отдельно в best-tuned campaign.

## Output Contract

Для прямого сравнения должны совпадать:

```text
codec and pixel format
NVENC preset and tuning
rate-control mode
target/min/max bitrate and VBV buffer
GOP and B-frames
FPS and frame count
limited BT.709 color metadata
MP4 container, no audio/subtitles/chapters
```

Canonical target: H.264 `yuv420p`, P4/HQ, B-frames 0, GOP в одну секунду,
35 Mbps для 1440p и 60 Mbps для 4K. Rate control явно зафиксирован как single-pass
CBR: target/min/max равны, VBV buffer равен двум секундам bitrate, initial
occupancy - одной секунде, lookahead и spatial/temporal AQ выключены.

Output валиден только после полного decode и проверки resolution, codec,
pixel format, color tags, FPS, duration, frame count, B-frames, keyframe interval,
фактического bitrate и монотонных PTS/DTS. Валидный MP4 после SHA256 может быть
удалён; невалидный сохраняется для диагностики.

## Quality Contract

Качество проверяется в двух точках:

1. Model-space parity: несколько RGB/float кадров до YUV conversion и encode.
2. Product-output parity: PSNR/SSIM и visual crops декодированных MP4.

Pixel diff только готовых MP4 недостаточен: он смешивает model output,
colorspace conversion и lossy encoder. VMAF/quality claims требуют отдельного
reference degradation dataset и не выводятся из throughput workload.

## Timing Contract

Основная метрика - full-process end-to-end FPS. Внешний monotonic timer включает
startup, decode, colorspace, inference, encode, flush и mux.

Для canonical run:

1. Отдельный discarded process обрабатывает 100 warmup frames.
2. Новый process обрабатывает ровно 1000 measured frames.
3. Выполняются минимум три run; при relative spread больше 5% - ещё два.
4. Порядок продуктов чередуется между раундами.
5. Между run выдерживается одинаковый idle interval.

Публикуются raw values, median, min/max и spread `(max - min) / median`.
Дополнительно фиксируются startup/context initialization, steady-state frame loop
и finalize/mux; эти scopes не заменяют full-process wall time. Cold-start и
warm-cache результаты не смешиваются.

Per-stage profiling и CUDA events являются diagnostics и выключены в измеряемом
hot path. Успешный smoke с уменьшенными параметрами получает `status: valid`, но
`publishable: false`.

## Metrics

Product/parity таблица содержит:

- median end-to-end FPS и wall time;
- average CPU utilization;
- average power и joules/frame;
- peak VRAM;
- output size и фактический bitrate.

`trtexec` публикуется отдельно. Диагностическая метрика:

```text
pipeline efficiency = ai-media-enhancer end-to-end FPS / trtexec QPS
```

Один representative run сопровождается Nsight Systems trace для проверки
H2D/D2H copies, stream gaps, CPU waits, PCIe traffic и overlap
NVDEC/TensorRT/NVENC. Trace не снимается внутри каждого measured run.

## Environment Contract

Все engines и сравнимые результаты строятся и измеряются на одной физической
GPU. Между сериями фиксируются driver, power limit, clocks, thermal policy,
Docker image digest и отсутствие display/посторонней GPU-нагрузки.

Runner сохраняет allowlisted environment:

- GPU model, compute capability, VRAM;
- CPU model и logical core count;
- driver, CUDA, TensorRT, CV-CUDA, PyNvVideoCodec, FFmpeg и Python versions;
- immutable image references и source revisions;
- power limit, clocks, temperature и throttle reasons;
- repository commit;
- SHA256 input, weights, ONNX, engine и sidecar;
- sanitized commands и benchmark parameters.

Не сохраняются hostname, username, IP, GPU UUID/serial, container IDs,
абсолютные host paths и полный environment dump.

## Validity And Success

Run недействителен при mismatch assets/contracts, output validation error,
посторонней GPU-нагрузке, thermal/hardware slowdown, изменении environment или
включённом per-frame profiler. Достижение заранее заданного SW power limit
фиксируется, но само по себе не инвалидирует run.

Критерий задаётся до получения результатов:

- больше 5% преимущества median FPS - подтверждённое преимущество;
- в пределах +/-5% - паритет; сравниваются CPU, energy/frame, VRAM и UX;
- больше 5% проигрыша на обоих workload - profiling и оптимизация до claim.

Individual suite всегда считается acceptance data, даже если использует
canonical frames/runs. Сравнительный результат формируется только rotated
campaign runner. До реализации CPU accounting, timing scopes и quality parity
даже валидная campaign получает `publishable: false`.
