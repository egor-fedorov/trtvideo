# Changes

Короткий журнал заметных проектных изменений. Это не замена `git log`: сюда
попадают изменения, которые важны для дальнейшей работы с проектом, workflow,
CLI, Docker, структурой файлов и эксплуатационными правилами.

Performance-изменения с цифрами и benchmark-сравнениями фиксируются отдельно в
`docs/PERFORMANCE_LOG.md`.

## Формат

Для новых записей использовать обратную хронологию:

```text
## YYYY-MM-DD - Краткое название

Что изменилось:

* ...

Цель:

* ...
```

## 2026-05-28 - Auto bitrate для NVENC backend

Что изменилось:

* `nvcodec` backend теперь по умолчанию оценивает target bitrate от source video bitrate;
* формула auto bitrate: `source_bitrate * (pixel_ratio * fps_ratio) ** 0.6`;
* `--bitrate-mbps` оставлен как явный override;
* `--crf` для `nvcodec` используется только как fallback, если bitrate исходника недоступен.

Цель:

* уменьшить риск случайно получить слишком большой output при smoke/batch прогонах;
* сохранить ручной контроль bitrate для воспроизводимых сравнений.

Проверки: `ruff`, `mypy`, `compileall`, CLI `--help`.

## 2026-05-27 - Переименование проекта и пакета

Что изменилось:

* проект переименован в `ai-media-enhancer`;
* Python package переименован с `upscaler` на `ai_media`;
* выбран root package layout без дополнительного слоя `src/`;
* Docker venv переименован в `/opt/ai-media-enhancer`;
* Docker image examples обновлены на `ai-media-enhancer:latest`;
* `CLAUDE.md` заменён на `AGENTS.md`;
* `OPTIMIZATIONS.md` перенесён в `docs/PERFORMANCE_LOG.md`;
* `TASKS.md` перенесён в `docs/archive/TASKS.md`;
* добавлен `docs/ROADMAP.md` как короткий актуальный план;
* удалён устаревший `scripts/run_batch.sh`.

Цель:

* название `ai-media-enhancer` лучше подходит под будущие video/image workflows;
* `ai_media` оставляет место для upscale, interpolation, restore и image tasks;
* документация разделена на актуальный контекст, roadmap, performance log и архив.

Проверки: `ruff`, `mypy`, `compileall`, CLI `--help`, `uv lock --check`, `git diff --check`.
