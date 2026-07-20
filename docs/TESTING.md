# Testing

Тесты проекта запускаются только через Docker dev image. Локальный host может не
иметь TensorRT, PyNvVideoCodec, CV-CUDA и совместимый Python runtime.

## Слои

### Unit

Быстрые pure-Python тесты без GPU/runtime-зависимостей.

```bash
make build-dev
make test-unit
```

Unit tests не должны импортировать TensorRT, CV-CUDA или PyNvVideoCodec.

### CLI/Docker Smoke

Будущий слой без GPU для проверки Docker image entrypoints:

```bash
docker run --rm ai-media-enhancer:dev upscale --help
docker run --rm ai-media-enhancer:dev benchmark-upscale --help
docker run --rm ai-media-enhancer:dev export-onnx --help
docker run --rm ai-media-enhancer:dev prepare-onnx --help
docker run --rm ai-media-enhancer:dev build-engine --help
```

### GPU Smoke

Будущий явный слой для GPU-хоста. Он должен использовать короткое synthetic video
и tiny TensorRT engine, а не реальные SPAN/RealESRGAN artifacts.

Проверять:

* output file существует;
* resolution соответствует scale;
* duration/frame count близки к expected;
* `pix_fmt` и color tags корректны;
* кадры не пустые и не зависают на первом кадре.

### Benchmark

Report-first слой пишет suite/run JSON, child logs и raw NVML samples без жёстких
FPS thresholds. Thresholds можно добавлять только после накопления baseline для
конкретных GPU, TensorRT version, backend, model и resolution.

Каноничные benchmark assets подготавливаются и проверяются без GPU:

```bash
make benchmark-prepare
make benchmark-verify
```

`benchmark-prepare` скачивает большие ignored assets и поэтому не входит в обычный
quality gate. Pure-Python контракты workload manifest и команд подготовки входят
в unit tests.

На GPU-хосте сначала выполняется короткая проверка runner/validation:

```bash
make build-benchmark
make benchmark-ai-media \
  BENCHMARK_VARIANT=720p \
  BENCHMARK_ENGINE=models/benchmarks/realesrgan-x2plus/engines/model.engine \
  BENCHMARK_ARGS="--runs 1 --extra-runs 0 --frames 120 --warmup-frames 24 --idle-seconds 0"
```

Полный 3+2 benchmark относится к Stage 2. Валидный run обязан пройти полный decode,
media/timestamp validation и NVML validity checks. `nvidia-ml-py` устанавливается
только в опциональный image `ai-media-enhancer:benchmark` и не входит в production
runtime.

## Quality Gate

Минимальный Docker gate для Python-изменений:

```bash
make build-dev
make check
```

`make check` не пересобирает dev image автоматически. После изменения зависимостей в
`pyproject.toml`/`uv.lock` или изменения `Dockerfile` сначала выполнить
`make build-dev`. Metadata-only изменение версии проекта пересборки image не требует.
