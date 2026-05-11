# Журнал Оптимизаций

Краткий журнал performance-изменений. Для каждой оптимизации фиксируем, что
изменилось, на каком benchmark проверялось, какой прирост получен и где он проявился.

## 2026-05-11 — NVDEC/NVENC buffer pool

Что изменено:

* добавлен per-job `FrameBufferPool` в `upscaler/gpu_pipeline.py`;
* `NV12->RGB` и `RGB->NV12` cvcuda conversion теперь пишут в preallocated buffers;
* `TensorRTRuntime.infer_rgb_tensor_into(...)` пишет output в preallocated RGB buffer;
* hot path `upscale-video-nvcodec` переиспользует `nv12_in`, `rgb_in`, `nchw_in`, `rgb_out`, `rgb_out_float`, `nv12_out`.

Benchmark:

* input: `videos/switzerland_720p.mp4`;
* engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_720p.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`;
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
* preprocess/postprocess path поддерживает FP16 bindings в `upscale-video` и `upscale-video-nvcodec`.

Benchmark:

* input: `videos/switzerland_1080p.mp4`;
* default engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p.engine`;
* FP16 I/O engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`;
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
* benchmark harness пробрасывает `--cuda-graph` в `upscale-video` и `upscale-video-nvcodec`;
* `TensorRTRuntime` пытается захватить TensorRT `execute_async_v3` в CUDA Graph;
* при ошибке capture runtime откатывается на обычный TensorRT enqueue.

Benchmark:

* input: `videos/switzerland_1080p.mp4`;
* engine: `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`;
* GPU: Quadro RTX 6000;
* command: `benchmark --model models/liveaction-span --backend nvcodec --engine-io-precision fp16 --cuda-graph --warmup-frames 20 --frames 1000`;
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
* Dockerfile устанавливает dependencies в venv `/opt/upscaler` с `--system-site-packages` до копирования application code;
* venv видит preinstalled packages из TensorRT base image, но не меняет managed `/usr`;
* uv download/wheel cache подключён через BuildKit cache mount;
* application code устанавливается быстрым `uv pip install --python "$VIRTUAL_ENV" --no-deps .`;
* `.dockerignore` исключает локальные cache dirs, benchmark JSON artifacts и временные логи.

Ожидаемый эффект:

* code-only изменения больше не должны переустанавливать `torch`, `cvcuda`,
  `pynvvideocodec`, `onnx`, `spandrel`;
* production image не должен содержать uv/pip download cache в финальном слое;
* фактическое ускорение нужно замерить на удалённом сервере через повторный
  `DOCKER_BUILDKIT=1 docker build -t upscaler:latest .` после изменения Python-кода.

## 2026-05-12 — Docker base image refresh

Что изменено:

* базовый образ обновлён с `nvcr.io/nvidia/tensorrt:26.03-py3` до
  `nvcr.io/nvidia/tensorrt:26.04-py3`;
* pipeline code не менялся, чтобы отдельно оценить влияние NVIDIA/TensorRT runtime stack.

Проверить на сервере:

* `DOCKER_BUILDKIT=1 docker build -t upscaler:latest .`;
* imports в контейнере: `tensorrt`, `torch`, `PyNvVideoCodec`, `cvcuda`, `spandrel`;
* `build-engine`;
* `upscale-video` и `upscale-video-nvcodec`;
* benchmark 720p/1080p и сравнение с baseline на `26.03-py3`.
