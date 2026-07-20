# Benchmarks

Этот каталог содержит воспроизводимый benchmark contract и небольшие текстовые
артефакты. Модели, ONNX, TensorRT engines, исходные и подготовленные видео в Git
не добавляются.

- `methodology.md` - правила измерения и сравнения реализаций.
- `workloads/` - входные данные для подготовки конкретного workload.
- `scripts/prepare_workload.py` - Docker-first подготовка и проверка assets.

Подготовка основного workload:

```bash
make build
make benchmark-prepare
make benchmark-verify
```

Первый запуск скачивает около 3.7 GB исходных данных Sintel. Результаты остаются
в игнорируемых каталогах `models/` и `videos/`.

Engine собирается на выбранной benchmark GPU перед снятием baseline. После сборки
соберите опциональный benchmark image и запустите короткий Stage 1 smoke:

```bash
make build-benchmark
make benchmark-ai-media \
  BENCHMARK_VARIANT=720p \
  BENCHMARK_ENGINE=models/benchmarks/realesrgan-x2plus/engines/model.engine \
  BENCHMARK_ARGS="--runs 1 --extra-runs 0 --frames 120 --warmup-frames 24 --idle-seconds 0"
```

Production image не содержит NVML dependency и benchmark runner scripts. Полный
baseline и competitor runners относятся к следующим этапам roadmap.

Повреждённые или несовместимые generated assets не перезаписываются молча. Для
явной повторной генерации используйте:

```bash
make benchmark-prepare BENCHMARK_ARGS=--force
```
