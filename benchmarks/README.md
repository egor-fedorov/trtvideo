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

Stage 0 не включает competitor runners и не создаёт TensorRT engine. Engine
собирается на выбранной benchmark GPU перед снятием baseline.

Повреждённые или несовместимые generated assets не перезаписываются молча. Для
явной повторной генерации используйте:

```bash
make benchmark-prepare BENCHMARK_ARGS=--force
```
