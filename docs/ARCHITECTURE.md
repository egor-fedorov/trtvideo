# Architecture

Этот документ описывает внутреннее устройство `ai-media-enhancer`. Публичные
команды запуска и подготовка моделей находятся в [`README.md`](../README.md),
тестовая стратегия - в [`TESTING.md`](TESTING.md), а результаты измерений
производительности - в [`PERFORMANCE_LOG.md`](PERFORMANCE_LOG.md).

## Область проекта

Проект предоставляет CLI-инструменты для AI-обработки медиа через TensorRT.
Сейчас реализован video upscale, но структура допускает добавление других media
workflow и runtime backend'ов.

Основные границы компонентов:

```text
ai_media/
  cli/          разбор аргументов и выбор команды/backend
  pipelines/    orchestration decode -> inference -> encode
  runtime/      TensorRT runtime и общий RuntimeEngine protocol
  video/        ffprobe metadata, FPS, bitrate и colorspace helpers
  models/       runtime-контракт модели (ModelSpec)
  profiling.py  сбор stage timings
```

CLI не ищет модели и engines автоматически. Пользователь явно передаёт static
TensorRT engine через `--engine`.

## Общий цикл инференса

Команда `upscale` строит общий parser, выбирает `--backend ffmpeg|nvcodec` и
передаёт готовые аргументы соответствующему pipeline. Общий жизненный цикл задаёт
`BasePipeline` в `ai_media/pipelines/base.py`:

1. Проверяет наличие `--engine` и `--input`.
2. Читает через `ffprobe` разрешение, FPS, количество кадров, bitrate и color
   metadata в `VideoInfo`.
3. Отклоняет входы за пределами текущего media contract.
4. Загружает указанный engine в `TensorRTRuntime` на выбранном `--gpu-id`.
5. Валидирует контракт модели и соответствие размера видео input shape engine.
6. Инициализирует decoder, encoder и переиспользуемые buffers выбранного backend.
7. Последовательно обрабатывает кадры и собирает статистику.
8. Flush-ит encoder, mux-ит результат при необходимости и освобождает ресурсы.
9. Печатает итоговую скорость и опциональный профиль.

`BasePipeline` владеет жизненным циклом и общей валидацией. Реализация decode,
preprocess, encode и cleanup остаётся в backend-классах.

## Контракт модели и TensorRT runtime

`TensorRTRuntime` в `ai_media/runtime/tensorrt.py`:

- десериализует TensorRT engine и создаёт execution context;
- читает имена, shapes и типы input/output tensors;
- строит `ModelSpec` и до выделения buffers проверяет, что engine представляет
  static single-frame RGB upscale с NCHW layout, batch 1 и равномерным integer
  scale;
- поддерживает FP32 и FP16 tensor bindings;
- заранее выделяет и переиспользует `gpu_input` и `gpu_output`;
- привязывает их к context через `set_tensor_address`;
- запускает inference через `execute_async_v3` на CUDA stream.

Runtime создаёт собственный `torch.cuda.Stream`. Caller может передать другой
stream и взять синхронизацию на себя. Это позволяет backend поместить preprocess,
TensorRT и postprocess в один упорядоченный GPU stream без host-side ожидания
между стадиями.

Экспериментальный `--cuda-graph` захватывает TensorRT enqueue для static-shape
engine. При ошибке capture runtime сохраняет причину и откатывается на обычный
`execute_async_v3`.

TensorRT engine зависит от версии TensorRT и GPU architecture. После смены
TensorRT container или класса GPU engine следует пересобрать. Sidecar
`<engine>.json` хранит metadata сборки, но не участвует в runtime discovery.

## Backend'ы

Оба backend используют TensorRT на GPU. Они отличаются способом decode, color
conversion, перемещения кадров и encode.

| Стадия | `ffmpeg` | `nvcodec` |
| --- | --- | --- |
| Decode | ffmpeg на CPU | NVDEC на GPU |
| Color conversion | ffmpeg/raw RGB и CPU buffers | CV-CUDA на GPU |
| TensorRT | GPU | GPU |
| Encode | `libx264` на CPU | NVENC на GPU |
| Копирование кадров через CPU | Да | Нет в основном data path |

### `ffmpeg` backend

Файл: `ai_media/pipelines/ffmpeg.py`.

```text
ffmpeg decode (CPU) -> RGB raw pipe -> numpy -> torch CUDA -> TensorRT
-> torch output -> CPU numpy -> RGB raw pipe -> libx264 encode (CPU)
```

Порядок обработки кадра:

1. Decoder subprocess пишет `rgb24` rawvideo в `stdout`.
2. Python читает один полный кадр и создаёт `numpy.ndarray [H, W, 3]`.
3. Runtime переносит RGB в CUDA, преобразует его в NCHW и нормализует в `0..1`.
4. TensorRT выполняет inference.
5. Output преобразуется в `uint8 RGB` и копируется на CPU.
6. Python пишет raw frame в `stdin` encoder subprocess.
7. ffmpeg кодирует видео через `libx264` и копирует аудиопоток исходника.

Backend прост по GPU-зависимостям, но CPU pipe и CPU codec добавляют копирования
и нагрузку. Качество управляется настоящим x264 `--crf`.

### `nvcodec` backend

Файл: `ai_media/pipelines/nvcodec.py`.

```text
NVDEC -> NV12 GPU surface -> CV-CUDA RGB -> TensorRT
-> CV-CUDA NV12 -> NVENC -> raw H.264/HEVC -> ffmpeg mux
```

Порядок обработки кадра:

1. `PyNvVideoCodec.ThreadedDecoder` декодирует compressed stream через NVDEC и
   выдаёт NV12 surface в device memory.
2. `torch.from_dlpack` получает GPU tensor без копирования кадра на CPU.
3. `FrameBufferPool` переиспользует заранее выделенные NV12, RGB и NCHW buffers.
4. CV-CUDA преобразует NV12 в RGB с явным SDR color spec.
5. RGB преобразуется в TensorRT input, после чего выполняется inference.
6. CV-CUDA преобразует output RGB обратно в NV12.
7. NV12 передаётся в NVENC через PyNvVideoCodec.
8. NVENC пишет raw H.264 или HEVC bitstream во временный файл.
9. В `finalize()` ffmpeg mux-ит video bitstream и опциональное аудио исходника в MP4.

В обычном non-profile path CV-CUDA, TensorRT и подготовка NV12 выполняются на
runtime CUDA stream. Тот же stream передаётся в NVENC через `cudastream`, поэтому
очередность GPU-операций сохраняется без `cudaStreamSynchronize` на каждом кадре.
CPU остаётся orchestration-слоем и пишет compressed bitstream, но не переносит
полные кадры между CPU и GPU.

NVENC работает без B-frames, сохраняет исходный rational FPS и создаёт GOP/IDR
примерно раз в секунду. Это обеспечивает монотонную временную разметку и пригодную
для перемотки структуру output. Качество задаётся явным `--bitrate-mbps` либо
автоматической оценкой от source bitrate:

```text
source_bitrate * (pixel_ratio * fps_ratio) ** 0.6
```

Auto bitrate является эвристикой. Для воспроизводимого размера файла следует
передавать bitrate явно.

## Media contract и цвет

Текущий video path рассчитан на SDR 8-bit input. Общая валидация отклоняет HDR
transfer functions, а `nvcodec` дополнительно принимает только `yuv420p`/`nv12`.
HDR, P010, YUV 4:2:2 и YUV 4:4:4 требуют отдельной color policy и tonemap.

Если source metadata отсутствует, pipeline использует безопасные SDR defaults:
BT.709 для HD/UHD и BT.601-compatible metadata для SD. NV12/RGB conversion в
CV-CUDA использует соответствующий явный color spec, а output получает заполненные
`color_range`, `color_space`, `color_transfer` и `color_primaries`.

При `--max-frames` output duration ограничивается по точному FPS, чтобы аудио не
продолжалось после последнего обработанного видеокадра.

## Профилирование и benchmark

`--profile` включает `ProfileCollector`. В `ffmpeg` backend измеряются:

- чтение decoded frame из pipe;
- CPU-to-GPU preprocess;
- TensorRT inference;
- GPU-to-CPU postprocess;
- запись кадра в encoder pipe.

В `nvcodec` backend измеряются:

- NV12 -> RGB через CV-CUDA;
- TensorRT inference;
- RGB -> NV12 через CV-CUDA;
- NVENC encode call.

GPU stages измеряются CUDA events. Profiled path может выполнять дополнительную
синхронизацию для получения корректных timing values, поэтому его CPU-поведение и
скорость не следует считать эквивалентными обычному inference path.

Текущий stage profiler начинает измерение после получения кадра от decoder и не
является полным end-to-end профилем процесса. `throughput_fps` в benchmark
считается по полному wall-clock времени backend run, а `processing_fps` - по
измеряемым кадрам без warmup.

`benchmark-upscale` запускает один или несколько backend на одном input/engine.
Он обрабатывает `warmup_frames + frames`, исключает warmup из processing metrics и
собирает JSON с FPS, frame timings, stage timings, GPU metadata, peak allocated
memory и состоянием CUDA Graph.

## Static и dynamic shapes

Текущий video inference path является static full-frame runtime: engine input shape
должен совпадать с разрешением входного видео.

Dynamic ONNX поддерживается на этапе сборки двумя способами:

1. `prepare-onnx` создаёт static variants для заданных разрешений.
2. `build-engine` принимает явные `--min-shape`, `--opt-shape` и `--max-shape`.

Dynamic engine с optimization profile можно собрать, но текущий runtime не выбирает
concrete shape и не перевыделяет buffers. Поэтому для `upscale` нужен static engine.

Для TensorRT 11 FP16 задаётся типами ONNX, а не weak-typing builder flags.
`prepare-onnx --precision fp16` переводит внутренние float tensors в FP16, сохраняя
FP32 I/O текущего video contract; затем `build-engine` компилирует подготовленный
ONNX без отдельного FP16-флага.

## Известные ограничения

- Реализован только full-frame video upscale с batch 1.
- Runtime не поддерживает dynamic-shape inference.
- Media contract ограничен SDR 8-bit video.
- Auto bitrate не гарантирует заданный размер или визуальное качество output.
- Stage profiler не включает полный decode-to-mux wall time для каждой стадии.
- TensorRT engines необходимо пересобирать при несовместимой смене TensorRT или GPU.
