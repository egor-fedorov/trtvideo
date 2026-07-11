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

* Добавлен каталог источников моделей `model_sources.json` и документация
  `docs/MODELS.md` с upstream links, лицензиями и локальными путями для RealESRGAN
  и `2xLiveActionV1_SPAN`.
* Добавлен короткий `run_span_batch.sh` для обработки всех видео из `./videos`
  через SPAN/NVENC.

### Changed
### Fixed
### Removed
```

В `CHANGES.md` попадают:

* изменения CLI, Docker workflow, model/engine registry и runtime defaults;
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
* `1.0.0` - когда CLI, Docker workflow и registry contract считаются стабильными.

## Unreleased

### Added
### Changed
### Fixed

* `nvcodec` backend теперь явно задаёт `gop` и `idrperiod` примерно в один
  ключевой кадр в секунду, чтобы output не получал один IDR/key frame на весь файл.

### Removed

* Удалён устаревший `RUNBOOK_REALESRGAN_SPAN.md`.
* Удалён устаревший архивный план `docs/archive/TASKS.md`.

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

### Цель

* Уменьшить риск случайно получить слишком большой output при smoke/batch прогонах.
* Сохранить ручной контроль bitrate для воспроизводимых сравнений.

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

### Цель

* Название `ai-media-enhancer` лучше подходит под будущие video/image workflows.
* `ai_media` оставляет место для upscale, interpolation, restore и image tasks.
* Документация разделена на актуальный контекст, roadmap, performance log и архив.
