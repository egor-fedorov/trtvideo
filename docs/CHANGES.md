# Changes

Versioned changelog для заметных пользовательских и эксплуатационных изменений.
Это не замена `git log` и не список каждого patch/refactor commit.

Performance-изменения с цифрами и benchmark-сравнениями фиксируются отдельно в
`docs/PERFORMANCE_LOG.md`.

## Как Вести

Новые заметные изменения добавлять в `Unreleased`, группируя по смыслу:

```text
## Unreleased

### Added
### Changed
### Fixed
### Removed
```

Версии располагать в обратном хронологическом порядке.

В `CHANGES.md` попадают:

* изменения CLI, Docker workflow, engine metadata и runtime defaults;
* изменения поведения output, encoding, color metadata, benchmark или manifest;
* миграции структуры проекта, влияющие на работу агента или разработчика;
* breaking changes и manual migration steps.

В `CHANGES.md` не попадают:

* мелкие refactor-only изменения без изменения поведения;
* вынесение magic numbers в константы;
* typo/docs cleanup без влияния на workflow;
* локальные runbook уточнения, если они не меняют интерфейс и runtime behavior.

## Версионирование

Версия в `pyproject.toml` поднимается на release, а не на каждый commit.

До `1.0.0` используем pragmatic semver:

* `0.1.PATCH` - bugfix, runbook/docs fix, совместимая эксплуатационная правка;
* `0.MINOR.0` - новая возможность, новый CLI/workflow, изменение default behavior;
* `1.0.0` - когда CLI и Docker/runtime workflow считаются стабильными.

## Unreleased

### Added

* Добавлен воспроизводимый Stage 0 benchmark contract для RealESRGAN_x2plus и
  Sintel, включая Docker-first команды `make -C benchmarks prepare` и
  `make -C benchmarks verify`, проверяемые source hashes и media/ONNX validation.
* `benchmark-upscale` переведён на внешний end-to-end timer, 3+2 run suite, NVML
  sampling, sanitized run manifests и автоматическую FFmpeg output validation.
* Добавлена каноничная команда `make -C benchmarks run-ai-media` для workload
  RealESRGAN_x2plus/Sintel.
* Benchmark runner и `nvidia-ml-py` изолированы в опциональном Docker target
  `benchmark`; production image не получает benchmark-only dependency и scripts.
* Добавлены зафиксированные Docker environments и отдельные runners для
  diagnostic `trtexec`, TensorRT 11 `vs-mlrt/vstrt` parity и stock
  `VSGAN-tensorrt-docker` product comparison с общей схемой результатов, NVML
  sampling, output validation и режимом `--dry-run` без GPU.
* Добавлен GPU benchmark runbook для будущей acceptance campaign на RTX 3090.
* Добавлен второй canonical workload на лёгкой `2xLiveActionV1_SPAN`: source
  hash, license/attribution, static ONNX variants и общие Sintel clips.
* Для stock VSGAN добавлена отдельная сборка TensorRT 10.16 engine из canonical
  ONNX с build log, sidecar contract и hashes. TRT11 engine проекта не
  переиспользуется между несовместимыми runtime versions.
* `export-onnx` получил явный `--name`, чтобы воспроизводимо экспортировать
  разные поддерживаемые Spandrel x2-модели без hardcoded RealESRGAN filename.

### Changed

* Canonical benchmark workload обновлён до `realesrgan-x2plus-sintel-v2`: общий
  H.264 input теперь не использует B-frames, что упрощает строгую проверку числа
  кадров и временной разметки; добавлен выборочный `prepare --force-clips` без
  пересборки ONNX.
* Per-stage profiling отделён от benchmark: stage timings доступны только через
  `upscale --profile/--profile-json`, а `benchmark-upscale` запускает обычный
  unprofiled pipeline отдельными warmup и measured процессами.
* Benchmark roadmap разделён на technical parity, stock product comparison и
  diagnostics. До публикационной campaign отдельно закрываются exact encoder
  contract, CPU/timing scopes, quality parity и чередование реализаций по run.
* Benchmark-specific Make targets перенесены в `benchmarks/Makefile`; корневой
  `Makefile` оставлен для build и quality gate основного проекта.

### Fixed

* `vstrt` runner передаёт абсолютный container path для input.
* При невалидном benchmark run конкретные manifest errors теперь сразу выводятся
  в stderr перед завершением Make target с кодом 2.
* Smoke overrides больше не могут ошибочно получить `publishable: true`: suite
  summary проверяет точное соответствие параметрам canonical workload.
* NVML process gate учитывает объявленную многопроцессную структуру внешних
  pipeline, сохраняя нулевой baseline для обнаружения посторонней GPU-нагрузки;
  повторяющиеся NVML records одного PID больше не считаются отдельными процессами.

### Removed

* Video2X удалён из canonical benchmark tooling: версия 6.4.0 выполняла
  `realesr-animevideov3`, а не используемую `RealESRGAN_x2plus`, поэтому её FPS
  нельзя использовать для same-model performance claim.

## 0.3.1 - 2026-07-20

### Changed

* `nvcodec` non-profile path теперь передаёт runtime CUDA stream в NVENC и не
  синхронизирует host thread перед каждым `Encode`, что снижает CPU busy-wait без
  изменения CLI.
* Описание inference, TensorRT runtime и backend'ов перенесено из агентских
  инструкций в публичный `docs/ARCHITECTURE.md`; `AGENTS.md` теперь содержит только
  правила работы агента и ссылки на каноническую документацию.

## 0.3.0 - 2026-07-12

### Changed

* Базовый Docker image обновлён до `nvcr.io/nvidia/tensorrt:26.06-py3`.
* TensorRT build workflow переведён на TensorRT 11 strong typing: FP16 теперь
  задаётся через `prepare-onnx --precision fp16`, а `build-engine` компилирует
  уже типизированный ONNX без precision builder flags.
* В runtime/export dependencies добавлен `onnxconverter-common` для lightweight
  mixed-precision ONNX graph rewrite.

### Fixed

* `nvcodec` backend теперь явно задаёт `gop` и `idrperiod` примерно в один
  ключевой кадр в секунду, чтобы output не получал один IDR/key frame на весь файл.
* `prepare-onnx --precision fp16` больше не запускает ModelOpt/ONNX Runtime reference
  pass на полном кадре и не требует 15+ GB памяти для 1080p conversion.

### Removed

* Удалены runtime model registry и automatic engine discovery. `upscale` и
  `benchmark-upscale` теперь требуют явный `--engine`; из `build-engine` удалён
  `--registry`. Sidecar `<engine>.json` остаётся метаданными конкретного engine.
* Удалён устаревший `RUNBOOK_REALESRGAN_SPAN.md`.
* Удалён устаревший архивный план `docs/archive/TASKS.md`.
* Из `build-engine` удалены weak-typing флаги `--fp16`, `--no-fp16` и
  experimental `--fp16-io`; для TensorRT 11 используйте FP16 ONNX.

## 0.2.0 - 2026-05-31

### Added

* Добавлен `Makefile` с Docker-only командами `build-dev`, `check`, `test-unit`,
  `lint`, `typecheck` и `compile`.
* Добавлена Docker-only unit test architecture на `pytest` для pure-Python контрактов.
* `nvcodec` backend теперь по умолчанию оценивает target bitrate от source video bitrate.
* Формула auto bitrate: `source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`.

### Changed

* `--bitrate-mbps` остаётся явным override для воспроизводимых прогонов.
* `--crf` больше не поддерживается в `nvcodec`; backend использует auto bitrate от
  source metadata или явный `--bitrate-mbps`. Если source bitrate недоступен,
  нужно явно передать `--bitrate-mbps`.

### Fixed

* `nvcodec` backend отключает B-frames в NVENC (`bf=0`), чтобы избежать reorder
  timestamps и ошибок вида `non monotonically increasing dts` при проверке MP4
  через ffmpeg.
* `nvcodec` backend больше не округляет дробный FPS до целого перед передачей в
  PyNvVideoCodec encoder; mux по-прежнему использует точный `ffprobe r_frame_rate`.

## 0.1.0 - 2026-05-27

### Changed

* Проект переименован в `ai-media-enhancer`.
* Python package переименован с `upscaler` на `ai_media`.
* Выбран root package layout без дополнительного слоя `src/`.
* Docker venv переименован в `/opt/ai-media-enhancer`.
* Docker image examples обновлены на `ai-media-enhancer:latest`.

### Docs

* `CLAUDE.md` заменён на `AGENTS.md`.
* `OPTIMIZATIONS.md` перенесён в `docs/PERFORMANCE_LOG.md`.
* `TASKS.md` перенесён в `docs/archive/TASKS.md`.
* Добавлен `docs/ROADMAP.md` как короткий актуальный план.
* Удалён устаревший `scripts/run_batch.sh`.
