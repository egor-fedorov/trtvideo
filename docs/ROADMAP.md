# Roadmap

Цель ближайшего цикла - проверить, имеет ли `ai-media-enhancer` измеримое
преимущество как NVIDIA/TensorRT video upscaler, и подготовить доказуемые
performance claims для open-source релиза.

Основной проверяемый тезис:

> GPU-resident pipeline `NVDEC -> CV-CUDA -> TensorRT -> CV-CUDA -> NVENC`
> обеспечивает высокий end-to-end throughput без выгрузки несжатых кадров на CPU.

## Stage 0. Benchmark Contract

Статус: выполнен. Методология и Docker-first подготовка
RealESRGAN_x2plus/Sintel workload зафиксированы в `benchmarks/`, assets успешно
прошли `make -C benchmarks verify` без GPU.

- Разделить три класса сравнения:
  - `trtexec` - inference ceiling для того же TensorRT engine;
  - `vs-mlrt/vstrt` - прямое сравнение TensorRT integration;
  - Video2X - product-level comparison, если одинаковая модель недоступна.
- Определить environment allowlist и правила фиксации GPU, driver, power limit,
  thermal policy, Docker image, версий конкурентов, hashes и точных команд.
- Определить границы end-to-end wall time, warmup policy и единый output contract:
  resolution, FPS, codec, bitrate/preset, audio и metadata.
- Основным workload сделать `1080p -> 4K`, дополнительным - `720p -> 1440p`.
- Выполнять минимум три measured run и публиковать median вместе с разбросом.
- Проверить лицензии benchmark-моделей и clips до добавления ссылок или данных в
  репозиторий.
- Публичный environment report строить по allowlist и не сохранять hostname,
  username, GPU UUID, абсолютные пути и другие данные конкретной системы.

## Stage 1. Measurement And Validation

Статус: реализовано, ожидает GPU acceptance smoke. Pure-Python contracts и Docker
workflow добавлены; performance results до Stage 3 на выбранной GPU не снимаются.

- Отделить machine-readable end-to-end metrics от per-stage profiling. Обычный
  benchmark не должен включать per-frame `torch.cuda.synchronize()` или другой
  instrumentation overhead.
- Считать основными межпродуктовыми метриками полное wall time и end-to-end FPS.
  Per-frame p50/p95 использовать только там, где определена сопоставимая latency
  семантика для асинхронных pipeline.
- Измерять total peak VRAM и power внешним NVML sampler, а не только через
  PyTorch allocator. `joules/frame` считать интегрированием power samples по
  времени measured run.
- Автоматизировать проверку GPU smoke output: полный decode через
  `ffmpeg -f null -`, FPS, duration, frame count, `has_b_frames`, keyframe
  interval и монотонность PTS/DTS в packet metadata.
- Добавить manifest одного запуска с environment, command, input/output hashes,
  metrics и результатом output validation.
- Использовать зафиксированный `benchmarks/methodology.md` как источник истины и
  добавить отдельные runner scripts. Общую campaign-команду добавлять только после
  стабилизации отдельных runners.

## Stage 2. Offline Competitor Tooling

Статус: выполнен offline. Версии и Docker environments зафиксированы, образы
собираются, unit tests и CLI dry-run проходят. Runtime compatibility и корректность
результатов остаются обязательным acceptance gate Stage 3.

- Зафиксировать версии, source revisions и отдельные Docker environments для
  `trtexec`, `vs-mlrt/vstrt` и Video2X, не добавляя их зависимости в production
  image проекта.
- Добавить отдельные runners `run_trtexec.py`, `run_vstrt.py` и
  `run_video2x.py`. Каждый runner должен читать canonical workload, формировать
  точную команду и сохранять результат в общей machine-readable схеме.
- Поддержать `--dry-run` и overrides для frames/runs, чтобы до аренды GPU проверить
  command generation, paths, mounts и manifests.
- Для `vstrt` подготовить один ONNX/tensor contract с проектом. Загрузку общего
  serialized engine считать предпочтительной, но не гарантировать до проверки
  TensorRT/plugin compatibility на GPU.
- Для Video2X подготовить запуск с теми же Real-ESRGAN x2 weights. Если точное
  совпадение model/runtime невозможно, заранее маркировать результат как
  product-level comparison.
- Переиспользовать единые правила external timing, NVML sampling и output
  validation, не встраивая profiler конкретного продукта в сравнимый hot path.
- Добавить unit tests для command generation/result parsing и Docker smoke
  `--help`/`--dry-run` без обязательного GPU.
- Добавить единый GPU runbook. Общую автоматизированную campaign-команду добавлять
  в Stage 3 после GPU smoke: она должна ротировать продукты между раундами, а не
  последовательно выполнять три независимых сгруппированных suite.

## Stage 3. RTX 3090 Acceptance And Baselines

Целевая карта первой benchmark campaign - одна физическая GeForce RTX 3090 с
24 GB VRAM. Все engines и сравнимые результаты собираются и снимаются в рамках
одного зафиксированного environment на этой карте.

- Зафиксировать driver, CUDA/TensorRT, power limit, clocks, thermal state, Docker
  image IDs и отсутствие display/посторонней GPU-нагрузки.
- Пересобрать TensorRT engines непосредственно на RTX 3090 и сохранить hashes
  ONNX, engine и sidecar manifest. Engines с других GPU/runtime не использовать.
- Сначала выполнить короткий smoke для `ai-media-enhancer`, `trtexec`, `vstrt` и
  Video2X на 720p, затем на основном `1080p -> 4K` workload. Проверить полный
  decode и output contract до длинных запусков.
- Исправлять обнаруженные runtime/ABI/model проблемы в рамках этой campaign, но
  после изменения кода, image или настроек заново запускать все затронутые
  сравнимые серии.
- После успешных smoke снять unprofiled baseline проекта и competitors: 1000
  кадров после warmup, минимум три запуска, ещё два при spread больше 5%.
- Ротировать порядок продуктов между раундами и не смешивать результаты до и
  после изменения environment.
- Запустить тот же engine через `trtexec` с точным числом iterations и рассчитать:

  ```text
  pipeline efficiency = ai-media-enhancer end-to-end FPS / trtexec FPS
  ```

- Не использовать `trtexec` как прямое продуктовое сравнение: он не выполняет
  decode, colorspace conversion, encode и mux.
- Для `vstrt` сравнить inference-only и полный video path; для Video2X отдельно
  проверить фактический bitrate, размер output, color metadata и визуальное
  качество.
- Quality comparison выполнять на лицензированном synthetic degradation dataset
  с lossless/reference outputs. PSNR, SSIM и VMAF дополнять visual crops и не
  смешивать model quality с потерями финального encoder.
- Сохранить sanitized raw results, точные команды и итоговую Markdown-таблицу в
  `benchmarks/results/<gpu-name>/`.

## Performance Claim Gate

Заявлять измеримое performance-преимущество, если выполняется хотя бы одно условие:

- end-to-end FPS выше `vstrt` минимум на 10% при одинаковой модели и настройках;
- end-to-end FPS выше Video2X минимум на 20% при сопоставимом качестве;
- скорость находится в пределах 5% от `vstrt`, но установка и запуск существенно
  проще и воспроизводятся одной Docker-командой.

Если проект проигрывает обоим решениям больше 10%, не публиковать заявления о
высокой производительности до Stage 4 и повторного benchmark. Этот gate ограничивает
performance claims, но сам по себе не запрещает open-source публикацию проекта.

## Stage 4. Measured Optimization

- Сначала снять timeline обычного non-profile path через Nsight Systems и
  подтвердить конкретные простои между NVDEC, CV-CUDA, TensorRT и NVENC.
- Проверить лишние GPU copies, HWC/NCHW transforms, FP16 conversion, блокировку в
  `NVENC Encode()` и влияние временного raw bitstream.
- При подтверждённых простоях прототипировать bounded ring buffer, double buffering,
  несколько CUDA streams и event dependencies для перекрытия `decode N+1`,
  `inference N` и `encode N-1`.
- Использовать Nsight Compute только для kernels, которые timeline определил как
  bottleneck; fused preprocess/postprocess рассматривать после измерения их доли.
- Отдельно проверить лёгкую модель, где orchestration и codec overhead занимают
  большую долю кадра, чем на SPAN/Real-ESRGAN.
- После каждого существенного изменения повторять один фиксированный benchmark,
  не меняя одновременно pipeline, модель и input.
- Фиксировать только измеренные результаты в `docs/PERFORMANCE_LOG.md`.

## Stage 5. Open-Source Release

- Добавить `LICENSE` и выполнить dependency/model/media license audit.
- Сделать английский README основным и добавить воспроизводимый one-command demo.
- Добавить CI для Ruff, mypy, pytest и Docker build без обязательного GPU.
- Добавить `CONTRIBUTING.md`, `SECURITY.md` и issue templates.
- Провести privacy/history audit репозитория перед публикацией.
- Исправить сохранение нескольких audio streams, subtitles, chapters и metadata
  либо явно задокументировать ограничения первого релиза.
- Опубликовать benchmark methodology, таблицу и sanitized raw results.
- Выпустить versioned GitHub Release.

## Later

- Улучшить media contract: VFR, rotation, SAR/DAR, duration и missing `nb_frames`.
- Расширить color/bit-depth поддержку за пределы SDR 8-bit NV12/yuv420p: P010,
  HDR metadata passthrough, tonemap и color management.
- Рассмотреть VapourSynth backend как отдельную продуктовую возможность, а не как
  обязательную часть competitor benchmark.
- Рассмотреть RIFE/frame interpolation после стабилизации video metadata и
  timestamp contract.
