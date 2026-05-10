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
