# Roadmap

Цель ближайшего цикла - проверить, имеет ли `ai-media-enhancer` измеримое
преимущество как NVIDIA/TensorRT video upscaler, и подготовить доказуемые
performance claims для open-source релиза.

Проверяемый архитектурный тезис:

> GPU-resident pipeline `NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC`
> обеспечивает высокий end-to-end throughput без выгрузки несжатых кадров на CPU.

## Benchmark Matrix

Результаты делятся на независимые классы:

1. Technical parity: `ai-media-enhancer` против локально собранного
   `VapourSynth/vstrt` на TensorRT 11 с одним serialized engine.
2. Stock product: `ai-media-enhancer` против pinned stock
   `VSGAN-tensorrt-docker` с одним ONNX, но отдельными native engines. Stock VSGAN
   использует TensorRT 10.16 и не может загрузить engine TensorRT 11.
3. Diagnostics: `trtexec`, stage profile и Nsight. Это не конкуренты и не строки
   продуктовой таблицы.

Video2X исключён: версия 6.4.0 не поддерживает используемую универсальную
`RealESRGAN_x2plus` и запускает другую anime-модель. Сравнение такого FPS не
проверяет эффективность pipeline.

Обязательные workload:

- `RealESRGAN_x2plus` - тяжёлый model-bound сценарий;
- `2xLiveActionV1_SPAN` - лёгкий сценарий, показывающий video pipeline overhead.

## Stage 0. Rebaseline Tooling

Статус: реализовано offline. `make check`, сборка benchmark/TRT11 vstrt images,
runner dry-run и static VSGAN Dockerfile check проходят. Полная сборка stock VSGAN
image, TensorRT engines и runtime smoke относятся к GPU acceptance Stage 1.

- Удалить Video2X из canonical tooling и документации.
- Перенести `trtexec` в diagnostic/reference класс.
- Добавить pinned stock VSGAN image без изменений его внутреннего кода.
- Сохранить строгий TRT11 `vstrt` runner для technical parity.
- Добавить SPAN workload с проверяемыми source hash, attribution и license.
- Для VSGAN собирать TRT10.16 engine из canonical mixed-FP16 ONNX на benchmark
  GPU и сохранять build log, engine sidecar и hashes.
- Зафиксировать критерии успеха и полный inference/output contract в
  `benchmarks/methodology.md`.

## Stage 1. GPU Acceptance

Целевая карта первой campaign - одна физическая GeForce RTX 3090 с 24 GB VRAM.

- Зафиксировать driver, power limit, clocks, thermal state и immutable image IDs.
- На этой GPU собрать TRT11 engines проекта и TRT10.16 engines stock VSGAN для
  RealESRGAN и SPAN. Engines из разных TensorRT runtime не переиспользовать.
- Проверить 720p smoke каждого runner: project, TRT11 `vstrt`, stock VSGAN и
  diagnostic `trtexec`. Затем повторить для 1080p.
- Для каждого video output проверить полный decode, frame count, timestamps,
  color tags, GOP/B-frames, bitrate и размер.
- Любое изменение image, engine, model или настроек инвалидирует затронутую
  серию и требует повторного smoke.

## Stage 2. Measurement Gaps

Статус: частично реализовано. Exact NVENC contract, rotated campaign runner и
sanitized acceptance-таблица готовы; остаются quality gates.

- [x] Явный одинаковый NVENC rate-control contract для проекта и VSGAN: codec,
  preset, tuning, RC mode, target/min/max bitrate, VBV, GOP и B-frames;
- [x] CPU utilization measured subprocess tree через `RUSAGE_CHILDREN`:
  user/system CPU seconds, average cores и affinity-normalized capacity;
- [x] Раздельные `startup`, steady-state frame loop и `finalize + mux` timing
  scopes с единым process/frame boundary contract;
- [ ] Model-space parity на RGB/float кадрах до YUV/encode;
- [ ] Product-output PSNR/SSIM и visual crops после декодирования MP4;
- [x] Campaign runner, который чередует продукты по раундам, а не запускает
  сгруппированные suite;
- [x] Генерация sanitized итоговой acceptance-таблицы из raw manifests.

Individual runner’ы остаются acceptance/baseline data. Campaign также не
публикуется до закрытия оставшихся Stage 2 gates.

## Stage 3. Parity Campaign

- Выполнить 100 warmup и 1000 measured frames, минимум три run и ещё два при
  spread больше 5%.
- Чередовать порядок project/vstrt/VSGAN между раундами.
- Сначала провести `1080p -> 4K`, затем confirmation `720p -> 1440p`.
- Повторить оба разрешения на RealESRGAN и SPAN.
- Публиковать median end-to-end FPS, wall time, CPU, average power,
  joules/frame, peak VRAM, output bitrate и размер.
- Сохранить команды, environment, raw values и hashes; не смешивать результаты
  разных commits, images или thermal/power state.

Критерий результата задаётся до измерений:

- преимущество больше 5% по median end-to-end FPS - подтверждённое преимущество;
- разница в пределах +/-5% - паритет, сравниваются CPU, energy/frame, VRAM и UX;
- проигрыш больше 5% на обоих workload - profiling и оптимизация до performance
  claim.

## Stage 4. Diagnostics And Best-Tuned

- Рассчитать `pipeline efficiency = end-to-end FPS / trtexec QPS` отдельно от
  продуктовой таблицы.
- Снять один representative Nsight Systems trace и проверить H2D/D2H copies,
  PCIe traffic, stream gaps, CPU waits и overlap NVDEC/TensorRT/NVENC.
- Выполнить отдельный best-tuned benchmark: VSGAN с рекомендованными requests,
  streams и CUDA Graph; проект - с лучшими подтверждёнными настройками.
- Оставить CUDA Graph experimental, пока захватывается только TensorRT call и
  нет измеримого выигрыша на текущих тяжёлых моделях.
- Повторить headline workload на коротком live-action clip с большим движением и
  мелкими деталями.
- Если timeline подтверждает простои, исследовать double buffering, несколько
  execution contexts и overlap `decode N+1`, `inference N`, `encode N-1`.
- После каждого изменения повторять один фиксированный benchmark и заносить в
  `docs/PERFORMANCE_LOG.md` только измеренный эффект.

## Stage 5. Open-Source Release

- Добавить `LICENSE` и выполнить dependency/model/media license audit.
- Сделать английский README основным и добавить one-command demo.
- Добавить CI для Ruff, mypy, pytest и Docker build без обязательного GPU.
- Добавить `CONTRIBUTING.md`, `SECURITY.md` и issue templates.
- Провести privacy/history audit репозитория.
- Исправить сохранение нескольких audio streams, subtitles, chapters и metadata
  либо явно задокументировать ограничения первого релиза.
- Опубликовать methodology, sanitized raw results и итоговые таблицы.
- Выпустить versioned GitHub Release.

## Later

- Улучшить media contract: VFR, rotation, SAR/DAR, duration и missing `nb_frames`.
- Добавить P010/HDR metadata passthrough, tonemap и color management.
- Рассмотреть VapourSynth backend как продуктовую возможность.
- Рассмотреть RIFE/frame interpolation после стабилизации video contract.
