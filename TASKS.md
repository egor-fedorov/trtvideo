# Task: укрепление ONNX/TensorRT video inference foundation и подготовка к VapourSynth/RIFE

## Контекст

Репозиторий сейчас является Docker-first CLI-утилитой для AI video upscaling через ONNX/TensorRT. Основные backend’ы:

* `ffmpeg pipe` backend: ffmpeg decode/encode + TensorRT inference.
* `NVDEC/NVENC + cvcuda + TensorRT` backend: GPU decode, GPU color conversion, TensorRT inference, GPU encode, затем mux через ffmpeg.

Текущий `NVDEC/NVENC + cvcuda + TensorRT` backend не нужно немедленно заменять на VapourSynth. Он ценен как production/full-GPU path с прямым контролем над decode, color conversion, inference, encode, profiling и будущей интеграцией в cloud/S3 worker.

VapourSynth/vs-mlrt/vstrt стоит добавить не как замену всей архитектуры, а как отдельный backend для graph-based video processing, RIFE/frame interpolation, model chaining и сравнения с VideoJaNai/Vapourkit.

## Общая стратегия

На первый этап добавить всё, что укрепляет текущую архитектуру и делает её пригодной не только для upscale, но и для разных ONNX video tasks.

На следующий этап добавить VapourSynth/vs-mlrt/vstrt и RIFE как отдельную ветку, когда базовые контракты уже будут выделены.

## Stage 0 — закрыть текущие эксплуатационные дыры

Этот этап должен идти перед большим архитектурным рефакторингом. Цель — сделать текущий Docker/ONNX/TensorRT workflow надёжным для уже используемых моделей и GPU.

Статус: реализовано в `prepare-onnx`, `build-engine`, README и CLAUDE. Runtime-проверку нужно выполнять в Docker, потому что локально нет ONNX/TensorRT зависимостей.

### 0.1. `prepare-onnx --size WIDTHxHEIGHT`

Сейчас `prepare_onnx` создаёт только встроенные static variants под 720p/1080p. Нужно добавить явное указание размера из CLI:

```bash
prepare-onnx models/onnx/model.onnx --size 1280x720
prepare-onnx models/onnx/model.onnx --size 1280x720 --size 1920x1080
```

Требования:

* `--size` принимает формат `WIDTHxHEIGHT`;
* если `--size` не указан, сохраняется текущее поведение: создаются 720p и 1080p variants;
* output filename должен включать понятный suffix, например `_720p`, `_1080p` или `_1280x720`;
* output shape вычисляется через `--scale`, как сейчас.

Definition of Done:

* `prepare-onnx model.onnx --size 1280x720` создаёт ONNX с input `[1, 3, 720, 1280]`;
* output shape становится `[1, 3, 1440, 2560]` при `--scale 2`;
* команда без `--size` продолжает создавать текущие 720p/1080p файлы.

### 0.2. Корректное распознавание dynamic ONNX dims

Сейчас dynamic detection учитывает только `dim_value == 0`, но TensorRT может показывать dynamic shape как `(-1, 3, -1, -1)`, а ONNX может хранить symbolic dims через `dim_param`.

Нужно считать dynamic:

* `dim_value == 0`;
* `dim_value < 0`;
* непустой `dim_param`.

Definition of Done:

* ONNX input `(-1, 3, -1, -1)` считается dynamic;
* symbolic dims вида `batch`, `height`, `width` считаются dynamic;
* static ONNX `[1, 3, 720, 1280]` не переписывается без необходимости.

### 0.3. Понятная ошибка `build-engine` для dynamic ONNX без profile

Сейчас dynamic ONNX падает внутри TensorRT с ошибкой `Network has dynamic or shape inputs, but no optimization profile has been defined`.

Нужно перед сборкой проверять network input shapes и печатать понятную ошибку до `build_serialized_network`, если ONNX dynamic и profile не задан.

Definition of Done:

* `build-engine dynamic.onnx` печатает сообщение, что ONNX dynamic и нужен либо `prepare-onnx`, либо explicit optimization profile;
* сообщение содержит пример команды для static workflow через `prepare-onnx`;
* TensorRT stacktrace/API Usage Error не является основным user-facing объяснением.

### 0.4. Минимальный dynamic-profile build

После понятной ошибки добавить минимальную поддержку TensorRT optimization profile:

```bash
build-engine model.onnx \
  --min-shape input:1x3x360x640 \
  --opt-shape input:1x3x720x1280 \
  --max-shape input:1x3x1080x1920
```

Definition of Done:

* dynamic ONNX собирается, если заданы `--min-shape`, `--opt-shape`, `--max-shape`;
* shape parser валидирует имя input tensor и rank;
* static ONNX продолжает собираться без profile;
* README и CLAUDE показывают оба workflow: static ONNX и dynamic ONNX with profile.

## Stage 1 — укрепить текущую архитектуру без переезда на VapourSynth

Stage 1 нужно выполнять после Stage 0 и не начинать с полного переписывания структуры проекта. Сначала выделить контракты, затем build/runtime foundation, затем performance work на основании benchmark.

### Stage 1A — Contracts

Цель: зафиксировать минимальные контракты для текущего single-frame upscale, не пытаясь сразу покрыть все будущие задачи.

Статус: реализован минимальный runtime contract без переноса проекта в новую структуру директорий.

### 1. Ввести минимальные `TensorSpec`, `VideoInfo`, `ModelSpec`

Сначала нужен небольшой набор контрактов, который описывает текущий upscale path:

```text
TensorSpec:
  name
  layout
  dtype
  shape
  pixel_format
  range

VideoInfo:
  width
  height
  fps
  nb_frames
  pix_fmt
  color metadata where available

ModelSpec:
  name
  task: upscale
  scale
  inputs
  outputs
  preprocess
  postprocess
```

Definition of Done:

* текущие SPAN/RealESRGAN-like модели описываются без дополнительных предположений в pipeline;
* runtime validation проверяет layout/range/scale для single-frame upscale;
* RIFE, matting, OCR, segmentation не входят в MVP `ModelSpec`, а остаются будущим расширением.

Текущая реализация:

* `TensorSpec` и `ModelSpec` описывают только static single-frame RGB upscale contract: NCHW, fp32, RGB, range `0_1`, batch=1;
* `TRTInference` создает `ModelSpec` при загрузке engine и валидирует static tensor shape, batch/channels и uniform integer scale до выделения GPU buffers;
* `VideoInfo` заменяет неструктурированный dict из `ffprobe`, но сохраняет dict-style доступ для существующих backend call sites;
* dynamic TensorRT runtime execution не входит в Stage 1A: текущий video inference path остается static-shape full-frame.
* локально проверено через `python3 -m py_compile`, `git diff --check` и smoke-check контрактов без TensorRT/GPU runtime.

### 2. Разделить CLI, runtime, video IO и task logic

Сейчас `BasePipeline` совмещает несколько ролей:

* CLI parsing;
* проверка input/output файлов;
* чтение video metadata;
* создание TensorRT runtime;
* проверка shape compatibility;
* decoder/encoder lifecycle;
* frame loop;
* profiling/statistics.

Нужно постепенно разнести это по слоям:

```text
cli/
  upscale.py
  benchmark.py
  inspect_model.py

core/
  model_spec.py
  runtime.py
  trt_runtime.py
  engine_cache.py
  task.py

video/
  info.py
  reader.py
  writer.py
  ffmpeg_pipe.py
  nvcodec.py
  colorspace.py

tasks/
  upscale.py
```

Цель: `upscale`, `RIFE`, `denoise`, `restore`, `background removal` и другие задачи должны быть разными task graph, а не копиями pipeline-классов.

### 3. Расширить `ModelSpec` после MVP

Сейчас код implicit-но предполагает модель вида:

```text
input:  [1, 3, H, W]
output: [1, 3, H*scale, W*scale]
layout: NCHW
pixel range: 0..1
task: single-frame upscale
```

Это подходит для SPAN/RealESRGAN-like upscale, но не подходит для RIFE, matting, segmentation, OCR, CLIP, denoise и temporal restoration.

Нужен явный `model.yaml` или `model.json` рядом с ONNX/engine.

Пример для upscale:

```yaml
name: 2xLiveActionV1-SPAN
task: upscale
runtime: tensorrt

inputs:
  - name: input
    layout: nchw
    dtype: fp32
    shape: [1, 3, dynamic, dynamic]
    pixel_format: rgb
    range: "0_1"

outputs:
  - name: output
    layout: nchw
    dtype: fp32
    pixel_format: rgb
    range: "0_1"

scale: 2
temporal: false
requires_padding_multiple: 8
supports_tiling: true
tile_overlap: 16

preprocess:
  colorspace: rgb
  normalize: "uint8_to_float_0_1"

postprocess:
  clamp: [0, 1]
  convert: "float_to_uint8"

license:
  commercial_use: false
  name: CC-BY-NC-SA-4.0
```

### Stage Q — Ruff/Mypy quality gates

Цель: сделать синтаксические и типовые проверки обязательной частью разработки, но отдельным этапом от Stage 1A.

Статус: реализовано как инкрементальный quality gate для текущего проекта.

Требования:

* добавить dev/development dependency group для `ruff` и `mypy`;
* привести `pyproject.toml` к реальной структуре этого проекта, без старых путей вроде `project/...`;
* зафиксировать команды проверки для локального и Docker-first workflow;
* прогнать `ruff check`, `mypy` и `python -m py_compile` по основным модулям;
* исправить найденные проблемы отдельным коммитом, не смешивая с архитектурными изменениями Stage 1A/1B.

Definition of Done:

* `ruff check .` проходит или имеет явно зафиксированный минимальный набор временных исключений;
* `mypy` проходит на выбранном scope проекта или имеет явно зафиксированный baseline;
* README/CLAUDE показывают команды quality checks;
* CI/local workflow может запускать эти проверки без ручной настройки.

Текущая реализация:

* добавлен optional extra `dev` с `ruff` и `mypy`;
* `pyproject.toml` очищен от старых путей `project/...` и чужих mypy overrides;
* `ruff check .` настроен на practical gate: `E`, `F`, `I`, `UP`, `B`, `SIM`, `W`;
* `mypy .` проверяет `inference.py`, `inference_gpu.py`, `tools`, `upscaler`;
* `mypy` временно подавляет `union-attr` до Stage 1B, потому что runtime fields сейчас инициализируются в `BasePipeline.run()`;
* Dockerfile поддерживает dev image через `--build-arg INSTALL_DEV=1`;
* локально проходят `ruff check .`, `mypy .`, `python3 -m compileall -q benchmark.py inference.py inference_gpu.py tools upscaler`.

### Stage 1B — Build/runtime foundation

Цель: отделить TensorRT runtime, engine metadata и build workflow от video pipeline.

Статус: реализован foundation с model/engine registry для static engines. Runtime
interface, sidecar engine manifest, TensorRT timing cache и автоматический подбор engine
по разрешению видео уже добавлены.

### 4. Ввести `RuntimeEngine` interface

Сейчас pipeline напрямую зависит от `TRTInference` и его внутренних полей: `gpu_input`, `gpu_output`, `stream`, `context`, `input_w`, `input_h` и т.д.

Нужно ввести интерфейс:

```python
class RuntimeEngine(Protocol):
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]

    def infer(
        self,
        inputs: dict[str, TensorLike],
        *,
        stream: CudaStream | None = None,
    ) -> dict[str, TensorLike]:
        ...
```

Реализации:

```text
TensorRTRuntime
ONNXRuntimeCudaRuntime     # optional/fallback later
VapourSynthRuntime         # later
```

Цель: VapourSynth/vs-mlrt можно будет добавить как backend, а не переписывать весь проект под него.

Definition of Done:

* video pipeline не обращается напрямую к `context`, `gpu_input`, `gpu_output`;
* TensorRT runtime остаётся первой и основной реализацией;
* stream ownership явно описан: caller может передать stream, иначе runtime использует свой.

Текущая реализация:

* добавлен `upscaler/runtime.py` с `RuntimeEngine` Protocol;
* `TRTInference` выделен в `TensorRTRuntime`, старое имя оставлено alias для совместимости;
* `FfmpegPipeline` и `GpuPipeline` больше не обращаются к `context`, `gpu_input`, `gpu_output`;
* CPU и GPU RGB inference идут через методы runtime: `infer_rgb_cpu`, `infer_rgb_cpu_profiled`, `infer_rgb_tensor`;
* caller может передать CUDA stream в runtime; если stream не передан, runtime использует собственный stream и синхронизируется сам.

### 5. Engine cache и engine manifest

Сейчас `.engine` указывается руками через `--engine`. Нужно перейти к model/engine registry.

Пример структуры:

```text
models/
  liveaction-span/
    model.onnx
    model.yaml
    engines/
      sm75_trt10.16_cuda12.6_fp16_1280x720.engine
      sm89_trt10.16_cuda12.6_fp16_1920x1080.engine
    manifest.json
```

Engine key должен учитывать:

```text
model_sha256
onnx_opset
tensorrt_version
cuda_version
gpu_compute_capability
precision
input_profile
builder_flags
preprocess_version
postprocess_version
```

Цель: избежать ситуации, когда engine существует, но собран под другую версию TensorRT, другой GPU class, другой shape или другие builder flags.

Текущая реализация:

* `build-engine` по умолчанию пишет sidecar manifest `<engine>.json`;
* manifest содержит ONNX hash, engine hash, TensorRT version, precision, input/output shapes, dynamic profile, builder flags, preprocess/postprocess versions и timing cache path;
* `--manifest PATH` задает путь manifest, `--no-manifest` отключает запись;
* `--registry PATH` обновляет model registry manifest, например `models/liveaction-span/manifest.json`;
* `upscale-video` и `upscale-video-nvcodec` поддерживают `--model PATH` и выбирают static engine по input resolution;
* `--engine PATH` остается прямым compatibility path;
* `--engine-precision fp16|fp32` ограничивает выбор engine из registry;
* registry пока выбирает только static-shape full-frame engines; dynamic runtime lookup остается будущей задачей.

### 6. Dynamic profiles и timing cache

Stage 0 добавляет минимальную сборку dynamic ONNX через явные `--min-shape`, `--opt-shape`, `--max-shape`.

Дальше нужно оставить static engine как fast path, расширить dynamic-profile build и добавить timing cache.

Пример CLI:

```bash
build-engine model.onnx \
  --min-shape input:1x3x360x640 \
  --opt-shape input:1x3x720x1280 \
  --max-shape input:1x3x1080x1920 \
  --fp16 \
  --timing-cache models/cache/trt.cache
```

Также полезны именованные профили:

```bash
build-engine model.onnx \
  --profile 720p:1x3x720x1280 \
  --profile 1080p:1x3x1080x1920 \
  --profile vertical:1x3x1280x720
```

Текущая реализация:

* добавлен `--timing-cache PATH`;
* existing static engine path сохранен как fast path;
* existing `--min-shape`, `--opt-shape`, `--max-shape` dynamic profile workflow сохранен;
* `--fp16` остается default behavior, `--no-fp16` позволяет собрать FP32 engine;
* именованные `--profile NAME:SHAPE` пока не реализованы.

### Stage 1C — Benchmark foundation

Цель: сначала получить машинно-читаемый benchmark, затем оптимизировать hot path по измерениям.

Статус: реализован базовый benchmark harness с JSON output.

### 7. Benchmark harness с JSON output

Нужна отдельная команда:

```bash
benchmark \
  --engine model.engine \
  --input video.mp4 \
  --backend ffmpeg,nvcodec \
  --warmup-frames 20 \
  --frames 300 \
  --json out.json
```

Метрики:

```json
{
  "model": "...",
  "engine": "...",
  "gpu": "...",
  "backend": "nvcodec",
  "input_resolution": "1280x720",
  "output_resolution": "2560x1440",
  "frames": 300,
  "fps_wall": 17.2,
  "stage_ms": {
    "decode": 2.1,
    "nv12_to_rgb": 0.7,
    "preprocess": 1.4,
    "trt": 52.0,
    "postprocess": 1.2,
    "rgb_to_nv12": 0.8,
    "encode": 3.0
  },
  "gpu_peak_mem_mb": 6200
}
```

Нужно сравнивать не только SPAN/heavy model, но и lightweight/compact модели. На тяжёлых моделях inference доминирует, поэтому NVDEC/NVENC и pipeline optimizations могут выглядеть менее значимыми.

Definition of Done:

* benchmark умеет запускать `ffmpeg` и `nvcodec` backend на одинаковом input/engine;
* JSON содержит FPS, stage timings, GPU name, input/output resolution, frame count;
* benchmark можно запускать в Docker без ручного парсинга текстовых логов.

Текущая реализация:

* добавлены команды `benchmark` и `benchmark-video`;
* benchmark запускает `upscale-video` и/или `upscale-video-nvcodec` как child process;
* `--engine` и `--model` оба поддержаны;
* `--json PATH` пишет JSON в файл, `--json -` пишет чистый JSON в stdout;
* progress и diagnostics benchmark выводятся в stderr, `--quiet` подавляет progress;
* pipeline получил `--profile-json PATH` и `--warmup-frames N`;
* итоговый JSON содержит backend, engine, GPU, input/output resolution, measured frames, warmup frames, `fps_wall`, `stage_ms`, `gpu_peak_mem_mb`;
* `ffmpeg` stage keys нормализованы в `decode`, `preprocess`, `trt`, `postprocess`, `encode`;
* `nvcodec` stage keys нормализованы в `nv12_to_rgb`, `trt`, `rgb_to_nv12`, `encode`;
* локально проверены quality gates и compile/smoke без TensorRT runtime.

### Stage 1D — Arbitrary resolution

### 8. Arbitrary resolution через padding/tiling

Сейчас input video size должен точно совпадать с engine input shape. Для реального cloud/S3 продукта это слишком жёстко.

Нужен режим:

```text
any input resolution
  -> pad to model multiple
  -> tile with overlap
  -> run model per tile
  -> crop overlap
  -> stitch output
  -> crop back to expected resolution
```

Пример config:

```python
@dataclass
class TilingConfig:
    enabled: bool
    tile_w: int = 512
    tile_h: int = 512
    overlap: int = 16
    pad_multiple: int = 8
    blend: Literal["none", "linear", "cosine"] = "linear"
```

Нужно поддерживать два режима:

```text
full-frame engine  -> fast path для известных разрешений
 tiled engine      -> fallback для любых разрешений и VRAM constraints
```

### Stage 1E — Pipeline performance

Эти задачи выполнять после benchmark harness, чтобы оптимизации были основаны на измерениях, а не на предположениях.

### 9. Buffer pool для GPU pipeline

В `NVDEC/NVENC + cvcuda + TensorRT` backend сейчас hot path создаёт GPU buffers для RGB/NV12 conversion на каждый кадр.

Нужно добавить preallocated buffer pool per job/shape:

```python
class FrameBufferPool:
    rgb_in: torch.Tensor
    nchw_in: torch.Tensor
    rgb_out: torch.Tensor
    nv12_out: torch.Tensor
```

Цель: уменьшить per-frame allocations, overhead и фрагментацию CUDA memory allocator.

### 10. Уменьшить per-frame synchronization

Сейчас `stream.synchronize()` вызывается на каждый кадр. Это упрощает корректность, но ограничивает throughput.

Нужно подготовить pipeline к:

```text
decode N+1 overlaps with inference N
inference N overlaps with encode N-1
```

Минимальный API:

```bash
--pipeline-depth 1|2|3
```

Внутри: ring buffer / double buffering / CUDA events вместо полной синхронизации после каждого кадра.

### 11. FP16 I/O benchmark

Сейчас TensorRT builder включает FP16 kernels, но runtime buffers могут оставаться FP32.

Нужно проверить режим:

```text
input tensor:  fp16
output tensor: fp16
preprocess: uint8 -> fp16 0..1
postprocess: fp16 -> uint8
```

CLI:

```bash
build-engine model.onnx --fp16 --fp16-io
```

Проверить:

* FPS;
* VRAM;
* визуальные артефакты;
* banding/clipping;
* совместимость моделей.

### 12. CUDA Graph benchmark для static-shape inference

Static engines подходят под CUDA Graph: fixed shape, fixed buffers, repeated command sequence.

Добавить experimental flag:

```bash
--cuda-graph
```

Цель: проверить снижение CPU launch overhead на compact/lightweight моделях.

### Stage 1F — Production media API cleanup

### 13. Улучшить video metadata layer

Сейчас достаточно width/height/fps/nb_frames, но для RIFE и production media processing этого мало.

Нужен `VideoInfo` dataclass:

```python
@dataclass
class VideoInfo:
    width: int
    height: int
    coded_width: int | None
    coded_height: int | None
    fps_nominal: Fraction
    avg_fps: Fraction | None
    time_base: Fraction
    duration_sec: float | None
    nb_frames: int | None
    pix_fmt: str
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    rotation: int | None
    is_vfr: bool
```

Важно для:

* VFR video;
* missing `nb_frames`;
* rotation metadata;
* SAR/DAR;
* HDR/color metadata;
* 10-bit formats;
* correct timestamps for RIFE.

### 14. Color/bit-depth support roadmap

Сейчас pipeline фактически 8-bit SDR-centric:

* ffmpeg backend: `rgb24` -> `libx264` -> `yuv420p`;
* nvcodec backend: NV12 path.

Нужно явно задокументировать уровни поддержки:

```text
Level 1: SDR 8-bit yuv420p/NV12
Level 2: SDR 10-bit P010
Level 3: HDR10 metadata passthrough
Level 4: HDR-aware processing / tonemap / colorspace management
```

На первом этапе минимум:

* сохранять color metadata там, где возможно;
* сохранять rotation metadata;
* явно писать, что текущий output — SDR 8-bit path;
* заложить P010 path как будущую задачу.

### 15. Encoder quality API

В GPU backend текущий `--crf` на самом деле мапится в estimated NVENC bitrate. Это нужно переименовать/расширить.

Предложение:

```bash
--encoder libx264|h264_nvenc|hevc_nvenc
--quality-mode crf|cq|bitrate
--crf 18
--cq 18
--bitrate 20M
--maxrate 30M
--bufsize 60M
--preset p1..p7
--tune hq|ll|ull|lossless
```

`--crf` оставить только для libx264/libx265 semantics. Для NVENC использовать `--cq` или `--bitrate`.

### 16. Machine-readable progress/profiling

Добавить:

```bash
--log-format text|json
--progress-file progress.jsonl
--profile-json profile.json
```

Пример событий:

```json
{"event": "job_started", "input": "...", "frames_total": 3504}
{"event": "frame_processed", "frame": 100, "fps": 17.2}
{"event": "stage_profile", "trt_ms": 52.1, "encode_ms": 3.0}
{"event": "job_finished", "output": "...", "duration_sec": 374.4}
```

Это нужно для будущей интеграции с worker/backend/S3/billing UI.

## Stage 2 — добавить VapourSynth/vs-mlrt/vstrt как отдельный backend

Не удалять текущий `NVDEC/NVENC + cvcuda + TensorRT` backend. Добавить VapourSynth рядом:

```text
backend=nvcodec       # production upscale fast path
backend=ffmpeg        # simple stable fallback
backend=vapoursynth   # graph/RIFE/experimental/model-chaining path
```

CLI варианты:

```bash
upscale-video --backend nvcodec ...
upscale-video --backend vapoursynth ...
interpolate-video --backend vapoursynth ...
```

Или отдельные команды:

```bash
upscale-video-nvcodec
upscale-video-vs
interpolate-video-vs
```

### Роль VapourSynth backend

Использовать для:

* RIFE/frame interpolation;
* chaining video filters;
* быстрого подключения чужих моделей;
* сравнения с VideoJaNai/Vapourkit;
* denoise/deband/sharpen/upscale chains;
* scene detection;
* advanced restoration graph.

### Минимальный `vapoursynth_pipeline.py`

Первый вариант может быть простым:

```text
input video
  -> generate temporary .vpy script
  -> VSPipe -c y4m script.vpy -
  -> ffmpeg encode/mux
```

Цель первого этапа VapourSynth backend — не сразу production zero-copy, а объективное сравнение скорости, качества, сложности кода и удобства RIFE/model chaining.

### Benchmark against native backend

Сравнить:

```text
A. current ffmpeg backend
B. current nvcodec backend
C. vapoursynth + vstrt backend
D. VideoJaNai/Vapourkit reference, optional
```

Тестовая матрица:

```text
1. 720p -> 1440p, SPAN / LiveAction
2. 1080p -> 4K, SPAN / LiveAction
3. 720p -> 1440p, lightweight compact model
4. 1080p -> 4K, lightweight compact model
5. RIFE 2x FPS, separate test
```

Метрики:

```text
- wall FPS
- GPU utilization
- CPU utilization
- VRAM peak
- RAM peak
- decode time
- inference time
- encode/mux time
- first-frame latency
- engine build/load time
- output visual correctness
```

Решение после benchmark:

```text
1. VapourSynth faster/same and much simpler -> possible default for some modes.
2. VapourSynth simpler but slower -> keep for RIFE/experimental graph path.
3. Native nvcodec clearly faster/more controllable -> keep as production upscale path.
```

## Stage 3 — RIFE/frame interpolation

RIFE нельзя воспринимать как ещё одну upscale ONNX-модель. Это temporal task:

```text
upscale:
  frame_t -> frame_t_upscaled

RIFE:
  frame_t + frame_t+1 + timestep -> intermediate_frame
```

Для RIFE нужны:

* frame pair reader;
* previous/current/next frame buffer;
* scene change detection;
* padding to model multiple;
* interpolation factor x2/x4/x8;
* timestamp generation;
* output frame ordering;
* VFR handling policy;
* duplicate/drop policy;
* optional audio passthrough/mux correctness.

### FrameProcessor abstraction

Добавить abstraction:

```python
class FrameProcessor(Protocol):
    temporal_radius: int

    def process(self, ctx: FrameContext) -> list[Frame]:
        ...
```

Для upscale:

```text
temporal_radius = 0
input: frame N
output: [upscaled frame N]
```

Для RIFE:

```text
temporal_radius = 1
input: frame N, frame N+1
output: [frame N, interpolated frame N+0.5]
```

Для будущих temporal restoration models:

```text
temporal_radius = 2
input: frame N-2..N+2
output: restored frame N
```

### RIFE ModelSpec example

```yaml
name: rife-v4.22
task: frame_interpolation
temporal: true

inputs:
  - name: frame0
    layout: nchw
    dtype: fp32
    range: "0_1"
  - name: frame1
    layout: nchw
    dtype: fp32
    range: "0_1"
  - name: timestep
    dtype: fp32
    shape: [1]

outputs:
  - name: middle_frame
    layout: nchw
    dtype: fp32
    range: "0_1"

requires_scene_detection: true
requires_padding_multiple: 32
```

## Что не делать сейчас

Не удалять текущий native `NVDEC/NVENC + cvcuda + TensorRT` backend.

Не делать полный переезд на VapourSynth до benchmark.

Не обещать поддержку “любых ONNX” без явного `ModelSpec`, shape/layout/range validation и engine compatibility checks.

Не встраивать VapourSynth так, чтобы весь проект стал зависеть от него архитектурно. Он должен быть backend’ом, а не фундаментом всего core.

## Предпочтительный порядок работ

1. Stage 0: закрыть текущие ONNX/TensorRT build проблемы.
2. Stage 1A: `VideoInfo`, `TensorSpec`, минимальный `ModelSpec` для single-frame upscale.
3. Stage Q: добавить `ruff`/`mypy` quality gates и поправить код по ним отдельным проходом.
4. Stage 1B: `RuntimeEngine` interface, TensorRT runtime cleanup, engine manifest/cache.
5. Stage 1B: dynamic profiles + timing cache.
6. Stage 1C: benchmark command + JSON output.
7. Stage 1D: arbitrary resolution через padding/tiling.
8. Stage 1E: buffer pool и частичное уменьшение synchronization в native GPU backend.
9. Stage 1E: FP16 I/O и CUDA Graph experiments на основании benchmark.
10. Stage 1F: encoder quality API cleanup и machine-readable progress/profiling.
11. Stage 2: добавить VapourSynth backend как experimental.
12. Stage 2: benchmark VapourSynth vs native backends.
13. Stage 3: добавить `interpolate-video-vs` для RIFE.

## Итоговое решение

На первый этап фокус нормальный: укрепить текущую архитектуру и сделать её пригодной для разных ONNX video tasks.

VapourSynth/RIFE лучше вынести на следующий этап. VapourSynth даст хороший задел под RIFE и graph-based processing, но текущий native `NVDEC/NVENC + cvcuda + TensorRT` backend стоит сохранить как production upscale fast path и benchmark baseline.
