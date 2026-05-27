# Performance Log

Исторический журнал performance-изменений. Актуальные команды запуска смотреть в
`README.md` и `AGENTS.md`.
Для новых performance-изменений фиксировать: что изменилось, benchmark/команду,
прирост или регресс и где он проявился.

## 2026-05-11 — NVDEC/NVENC buffer pool

Что изменено:

* добавлен per-job `FrameBufferPool` в `ai_media/pipelines/nvcodec.py`;
* `NV12->RGB` и `RGB->NV12` cvcuda conversion теперь пишут в preallocated buffers;
* `TensorRTRuntime.infer_rgb_tensor_into(...)` пишет output в preallocated RGB buffer;
* hot path `upscale --backend nvcodec` переиспользует `nv12_in`, `rgb_in`, `nchw_in`, `rgb_out`, `rgb_out_float`, `nv12_out`.

Benchmark:

* input: `videos/switzerland_720p.mp4`;
* engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_720p.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark-upscale --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`;
* source artifact: `switzerland_720p_benchmark.json` (локальный benchmark artifact, не коммитится).

Результат:

| Backend | Метрика | До | После | Изменение |
| --- | ---: | ---: | ---: | ---: |
| `nvcodec` | processing FPS | 36.5 | 37.44 | +2.6% |
| `nvcodec` | avg frame time | 27.4 ms | 26.71 ms | -2.5% |
| `nvcodec` | `NV12->RGB` stage | 0.8 ms | 0.50 ms | -37.5% |
| `nvcodec` | `TRT inference` stage | 25.7 ms | 25.49 ms | -0.8% |

Вывод:

* прирост виден в cvcuda stage `NV12->RGB`;
* общий FPS вырос умеренно, потому что TensorRT inference всё ещё доминирует в 720p SPAN run;
* `ffmpeg` backend тоже измеряется benchmark'ом, но эта оптимизация не была направлена на `ffmpeg`;
* для end-to-end скорости использовать `throughput_fps`, для hot-path анализа — `processing_fps` и `stage_ms`.

## 2026-05-11 — FP16 I/O experiment

Что изменено:

* добавлен experimental `build-engine --fp16-io`;
* TensorRT input/output bindings можно собрать как FP16 вместо FP32;
* registry selection поддерживает `--engine-io-precision fp16|fp32`;
* runtime выделяет input/output buffers по dtype engine bindings;
* preprocess/postprocess path поддерживает FP16 bindings в `upscale --backend ffmpeg` и `upscale --backend nvcodec`.

Benchmark:

* input: `videos/switzerland_1080p.mp4`;
* default engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p.engine`;
* FP16 I/O engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark-upscale --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`;
* FP16 I/O command adds: `--engine-io-precision fp16`;
* source artifacts: `switzerland_1080p_default_benchmark.json`, `switzerland_1080p_fp16io_benchmark.json` (локальные benchmark artifacts, не коммитятся).

Результат:

| Backend | Метрика | Default FP16 / FP32 I/O | FP16 I/O | Изменение |
| --- | ---: | ---: | ---: | ---: |
| `ffmpeg` | throughput FPS | 7.61 | 7.84 | +3.0% |
| `ffmpeg` | processing FPS | 16.27 | 16.66 | +2.4% |
| `ffmpeg` | avg frame time | 61.48 ms | 60.02 ms | -2.4% |
| `ffmpeg` | GPU peak memory | 261.84 MB | 143.87 MB | -45.1% |
| `ffmpeg` | postprocess stage | 1.60 ms | 0.94 ms | -41.3% |
| `ffmpeg` | TRT stage | 52.80 ms | 52.07 ms | -1.4% |
| `nvcodec` | throughput FPS | 16.53 | 17.20 | +4.1% |
| `nvcodec` | processing FPS | 16.80 | 17.51 | +4.2% |
| `nvcodec` | avg frame time | 59.51 ms | 57.10 ms | -4.1% |
| `nvcodec` | GPU peak memory | 282.74 MB | 164.90 MB | -41.7% |
| `nvcodec` | `TRT inference` stage | 58.19 ms | 55.71 ms | -4.3% |

Вывод:

* FP16 I/O даёт умеренный speedup на 1080p -> 4K, заметнее на `nvcodec`;
* главный эффект — снижение peak GPU memory примерно на 42-45%;
* для 1080p SPAN inference всё ещё доминирует TensorRT compute;
* перед production default нужно отдельно проверить визуальные артефакты, banding/clipping и совместимость других моделей.

Замечание по Docker mount:

* команда с `-v "$PWD:/app/artefacts"` и `--json artefacts/name.json` сохраняет файл в host `$PWD/name.json`, потому что внутри контейнера относительный путь `artefacts/name.json` резолвится как `/app/artefacts/name.json`;
* для сохранения в host `./artefacts/name.json` использовать `-v "$PWD/artefacts:/app/artefacts"` или монтировать весь repo как `-v "$PWD:/app"`.

## 2026-05-11 — CUDA Graph experiment

Что изменено:

* добавлен experimental `--cuda-graph`;
* benchmark harness пробрасывает `--cuda-graph` в `upscale --backend ffmpeg|nvcodec`;
* `TensorRTRuntime` пытается захватить TensorRT `execute_async_v3` в CUDA Graph;
* при ошибке capture runtime откатывается на обычный TensorRT enqueue.

Benchmark:

* input: `videos/switzerland_1080p.mp4`;
* engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark-upscale --model models/liveaction-span --backend nvcodec --engine-io-precision fp16 --cuda-graph --warmup-frames 20 --frames 1000`;
* source artifacts: `switzerland_1080p_fp16io_benchmark.json`, `switzerland_1080p_fp16io_cuda_graph_benchmark.json` (локальные benchmark artifacts, не коммитятся).

Результат:

| Backend | Метрика | FP16 I/O | FP16 I/O + CUDA Graph | Изменение |
| --- | ---: | ---: | ---: | ---: |
| `nvcodec` | `cuda_graph` | false | true | capture работает |
| `nvcodec` | processing FPS | 17.51 | 17.86 | +2.0% |
| `nvcodec` | throughput FPS | 17.20 | 17.15 | -0.3% |
| `nvcodec` | avg frame time | 57.10 ms | 55.98 ms | -2.0% |
| `nvcodec` | `TRT inference` stage | 55.71 ms | 54.47 ms | -2.2% |
| `nvcodec` | GPU peak memory | 164.90 MB | 164.90 MB | без изменений |

Вывод:

* CUDA Graph capture теперь реально включается: `cuda_graph: true`, `cuda_graph_error: null`;
* на тяжёлой 1080p SPAN-модели эффект небольшой, потому что время кадра dominated by TensorRT compute;
* end-to-end throughput фактически в пределах шума, поэтому production default пока не менять;
* следующий полезный замер — lightweight/compact model, где CPU launch overhead должен быть заметнее.

## 2026-05-11 — Docker dependency layer/cache

Что изменено:

* Docker runtime dependencies читаются из `pyproject.toml`/`uv.lock` через `uv export --frozen --no-emit-project`;
* dev-only dependencies оформлены как `[dependency-groups].dev` и ставятся через `--group dev` при `--build-arg INSTALL_DEV=1`;
* Dockerfile устанавливает dependencies в venv `/opt/ai-media-enhancer` с `--system-site-packages` до копирования application code;
* venv видит preinstalled packages из TensorRT base image, но не меняет managed `/usr`;
* uv download/wheel cache подключён через BuildKit cache mount;
* application code устанавливается быстрым `uv pip install --python "$VIRTUAL_ENV" --no-deps .`;
* `.dockerignore` исключает локальные cache dirs, benchmark JSON artifacts и временные логи.

Подтверждённый эффект:

* warm-cache rebuild после code-only изменения переиспользует dependency layer;
* тяжёлые зависимости больше не переустанавливаются: `torch`, `cvcuda`,
  `pynvvideocodec`, `onnx`, `onnxscript`, `spandrel`;
* production image не должен содержать uv/pip download cache в финальном слое;
* точное время ускорения зависит от host cache и Docker storage driver, поэтому
  конкретные секунды фиксировать только вместе с полным build log.

## 2026-05-12 — Docker base image refresh

Что изменено:

* базовый образ обновлён с `nvcr.io/nvidia/tensorrt:26.03-py3` до
  `nvcr.io/nvidia/tensorrt:26.04-py3`;
* pipeline code не менялся;
* `26.04` benchmark выполнялся на engine, пересобранном на новом образе, поэтому это
  сравнение полного TensorRT stack refresh, а не чистый runtime-only A/B на одном и том
  же `.engine`.

Benchmark:

* input: `videos/switzerland_720p.mp4`, `videos/switzerland_1080p.mp4`;
* engine: FP16 I/O engines from `models/liveaction-span/engines/`, для `26.04` engine
  был пересобран на новом TensorRT runtime;
* GPU: Quadro RTX 6000;
* command: `benchmark-upscale --model models/liveaction-span --backend nvcodec --engine-io-precision fp16 --warmup-frames 20 --frames 1000`;
* CUDA Graph command adds: `--cuda-graph`;
* source artifacts: `artefacts/switzerland_720_2604_fp16io_benchmark.json`,
  `artefacts/switzerland_720p_2604_fp16io_cuda_graph_benchmark.json`,
  `artefacts/switzerland_1080p_2604_fp16io_benchmark.json`,
  `artefacts/switzerland_1080p_2604_fp16io_benchmark_2.json`,
  `artefacts/switzerland_1080p_2604_fp16io_cuda_graph_benchmark.json`.
* для `1080p` без CUDA Graph в колонке `26.04` указано среднее двух прогонов.

Сравнение `26.03` stack -> `26.04` stack без CUDA Graph:

| Input | Метрика | `26.03` FP16 I/O | `26.04` FP16 I/O | Изменение |
| --- | ---: | ---: | ---: | ---: |
| 720p | processing FPS | 37.76 | 37.22 | -1.4% |
| 720p | throughput FPS | 36.75 | 35.81 | -2.5% |
| 720p | avg frame time | 26.49 ms | 26.87 ms | +1.4% |
| 720p | `TRT inference` stage | 25.29 ms | 25.00 ms | -1.1% |
| 1080p | processing FPS | 17.51 | 16.78 | -4.2% |
| 1080p | throughput FPS | 17.20 | 15.93 | -7.4% |
| 1080p | avg frame time | 57.10 ms | 59.58 ms | +4.3% |
| 1080p | `TRT inference` stage | 55.71 ms | 57.64 ms | +3.5% |

CUDA Graph на `26.04`:

| Input | Метрика | FP16 I/O | FP16 I/O + CUDA Graph | Изменение |
| --- | ---: | ---: | ---: | ---: |
| 720p | `cuda_graph` | false | true | capture работает |
| 720p | processing FPS | 37.22 | 36.99 | -0.6% |
| 720p | throughput FPS | 35.81 | 34.09 | -4.8% |
| 720p | avg frame time | 26.87 ms | 27.04 ms | +0.6% |
| 720p | `TRT inference` stage | 25.00 ms | 24.97 ms | -0.1% |
| 1080p | `cuda_graph` | false | true | capture работает |
| 1080p | processing FPS | 16.78 | 16.14 | -3.8% |
| 1080p | throughput FPS | 15.93 | 15.72 | -1.3% |
| 1080p | avg frame time | 59.58 ms | 61.94 ms | +4.0% |
| 1080p | `TRT inference` stage | 57.64 ms | 59.95 ms | +4.0% |

Вывод:

* переход на `26.04` вместе с пересборкой engine не дал speedup на текущей SPAN FP16 I/O модели и Quadro RTX 6000;
* 720p почти в пределах шума, но 1080p стал медленнее по processing FPS и TRT stage;
* CUDA Graph capture работает на `26.04`, но не даёт устойчивого выигрыша на 720p/1080p SPAN;
* `--cuda-graph` не переводить в default, оставить experimental flag;
* `build-engine` на новом образе фактически проверен пересборкой engine;
* если нужен чистый runtime-only A/B, нужно отдельно прогнать старый `26.03` engine внутри `26.04` image, но такой замер может быть нерепрезентативен из-за совместимости TensorRT engine/runtime.
