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

* Добавлен `Makefile` с Docker-only командами `build-dev`, `check`, `test-unit`,
  `lint`, `typecheck` и `compile`.
* Добавлена Docker-only unit test architecture на `pytest` для pure-Python контрактов.
* `nvcodec` backend теперь по умолчанию оценивает target bitrate от source video bitrate.
* Формула auto bitrate: `source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`.

### Changed

* `--bitrate-mbps` остаётся явным override для воспроизводимых прогонов.
* `--crf` для `nvcodec` используется только как fallback, если bitrate исходника недоступен.

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
