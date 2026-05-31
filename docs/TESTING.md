# Testing

Тесты проекта запускаются только через Docker dev image. Локальный host может не
иметь TensorRT, PyNvVideoCodec, CV-CUDA и совместимый Python runtime.

## Слои

### Unit

Быстрые pure-Python тесты без GPU/runtime-зависимостей. Это единственный тестовый
слой, который сейчас реализуется.

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

Будущий report-first слой. На первом этапе benchmark пишет JSON artifact без
жёстких FPS thresholds. Thresholds можно добавлять только после накопления baseline
для конкретных GPU, TensorRT version, backend, model и resolution.

## Quality Gate

Минимальный Docker gate для Python-изменений:

```bash
make build-dev
make check
```

`make check` не пересобирает dev image автоматически. После изменения `pyproject.toml`,
`uv.lock` или `Dockerfile` сначала выполнить `make build-dev`.
