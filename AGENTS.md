# AI Media Enhancer - правила работы агента

## Контекст проекта

`ai-media-enhancer` - Docker-first CLI-инструменты для AI-обработки медиа через
TensorRT. Текущий реализованный workflow - апскейл видео через `ffmpeg` или
`NVDEC/NVENC` backend.

Production runtime использует Python 3.12 из базового TensorRT Docker image
`nvcr.io/nvidia/tensorrt:26.06-py3`. Разработка ведётся локально, а проверки с
TensorRT, PyNvVideoCodec, CV-CUDA и GPU обычно выполняются в Docker на удалённом
GPU-хосте.

## Источники истины

- `README.md` - публичный Docker workflow, CLI и подготовка моделей.
- `docs/ARCHITECTURE.md` - устройство inference, runtime и backend'ов.
- `docs/TESTING.md` - тестовые слои и Docker-only quality gate.
- `docs/ROADMAP.md` - короткий актуальный план.
- `docs/CHANGES.md` - заметные изменения и правила версионирования.
- `docs/PERFORMANCE_LOG.md` - измеренные performance-изменения.

Не дублировать в этом файле пользовательскую или архитектурную документацию.
При изменении поведения обновлять соответствующий канонический документ.

## Правила работы

- Основной workflow - Docker-first. Не считать локальное отсутствие runtime-only
  зависимостей ошибкой проекта.
- Не коммитить `models/`, `videos/` и большие runtime artefacts без явной команды.
- Веса, ONNX и TensorRT engines не vendored в репозиторий.
- Перед полным batch-прогоном сначала делать короткий smoke через `--max-frames`.
- Для изменений в color/encoding path проверять не только запуск, но и `ffprobe`:
  `pix_fmt`, `color_range`, `color_space`, `color_transfer`, `color_primaries`,
  bitrate, duration, frame count и временную разметку кадров.
- Заметные изменения workflow, CLI, Docker, структуры файлов и проектных правил
  фиксировать в `docs/CHANGES.md`.
- Performance-изменения фиксировать в `docs/PERFORMANCE_LOG.md` только вместе с
  измерением: что изменилось, какой benchmark использован, какой получен прирост
  или регресс.
- Если проверку нельзя выполнить локально из-за отсутствующих GPU/runtime
  зависимостей, явно указывать это в итоговом ответе.
- Unit tests должны оставаться pure-Python и не импортировать TensorRT, CV-CUDA
  или PyNvVideoCodec.
- Не нарушать Docker dependency cache без необходимости: dependency metadata
  копируется до application code, а проект устанавливается отдельным слоем.

## Проверки

Проверочные инструменты устанавливаются в Docker dev image:

```bash
make build-dev
make check
```

После Python-изменений минимум запускать `ruff check .` через dev image. Перед
коммитом Python-кода запускать полный `make check`: `ruff`, `mypy`, `compileall`
и unit tests.

GPU/runtime smoke и benchmark выполняются на GPU-хосте. Команды и критерии
проверки описаны в `README.md` и `docs/TESTING.md`.
