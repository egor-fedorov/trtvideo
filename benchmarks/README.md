# Benchmarks

Каталог содержит воспроизводимый benchmark contract, изолированные competitor
images и runners. Модели, ONNX, TensorRT engines, исходные видео и raw results в
Git не добавляются.

- `methodology.md` - правила измерения и сравнения.
- `workloads/` - manifests каноничных workload.
- `competitors.json` - зафиксированные версии и классы сравнения.
- `docker/` - отдельные environments для `vs-mlrt/vstrt` и Video2X.
- `scripts/` - подготовка assets и runners.
- `GPU_RUNBOOK.md` - последовательность запуска на benchmark GPU.

Benchmark workflow отделён от основного `Makefile`:

```bash
make -C benchmarks help
```

## Assets

Сначала соберите production image, затем подготовьте и проверьте workload:

```bash
make build
make -C benchmarks prepare
make -C benchmarks verify
```

Первый запуск скачивает около 3.7 GB исходных данных Sintel. Результаты остаются
в игнорируемых каталогах `models/` и `videos/`. Повторная генерация несовместимых
assets выполняется явно:

```bash
make -C benchmarks prepare ARGS=--force
```

## Offline Gate

Соберите опциональный benchmark image и образы конкурентов:

```bash
make -C benchmarks build-vstrt
make -C benchmarks build-video2x
```

Проверить command generation без GPU и без готового engine:

```bash
make -C benchmarks dry-run \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250" \
  VSTRT_ARGS="--warmup-frames 24" \
  VIDEO2X_ARGS="--warmup-frames 24"
```

`trtexec` измеряет inference ceiling того же TensorRT engine. `vstrt` является
прямым TensorRT-сравнением и получает тот же engine. Stock Video2X 6.4.0 не
содержит RealESRGAN_x2plus и запускается с `realesr-animevideov3` x2, поэтому его
результат всегда маркируется как product-level comparison. Software decode для
него задан явно: stock RealESRGAN preprocessing Video2X 6.4.0 не принимает CUDA
AVFrames, хотя inference и encode продолжают выполняться на GPU.

Параметры `--frames`, `--warmup-frames`, `--runs`, `--extra-runs` и
`--idle-seconds` можно уменьшать для smoke-проверок. Успешный smoke получает
`status: valid`, но `publishable: false`: публикуемым считается только suite,
полностью совпадающий с параметрами workload manifest.

Production image не содержит NVML dependency и competitor tools. Фактическая
GPU-проверка и полные 3+2 серии описаны в `GPU_RUNBOOK.md`.
