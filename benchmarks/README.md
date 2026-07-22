# Benchmarks

Каталог содержит воспроизводимые workload manifests, pinned implementation
metadata, изолированные Docker environments и runners. Модели, ONNX, TensorRT
engines, исходные видео и raw results в Git не добавляются.

- `methodology.md` - классы сравнения и критерии валидности.
- `workloads/` - RealESRGAN и SPAN workload manifests.
- `implementations.json` - pinned diagnostic/parity/product implementations.
- `docker/` - TensorRT 11 vstrt parity и stock VSGAN environments.
- `scripts/` - подготовка assets, engine builder и runners.
- `GPU_RUNBOOK.md` - последовательность acceptance на benchmark GPU.

Benchmark workflow отделён от корневого `Makefile`:

```bash
make -C benchmarks help
```

## Матрица

- `run-vstrt` - technical parity на том же TensorRT 11 engine.
- `run-vsgan` - stock product comparison на том же ONNX, но отдельном TRT10.16
  engine из-за несовместимости serialized engines между runtime versions.
- `run-trtexec` - diagnostic inference ceiling, не конкурент.

Video2X исключён: он не выполнял canonical `RealESRGAN_x2plus`, поэтому его FPS
не отвечал на вопрос о производительности одинаковой модели.

## Assets

RealESRGAN является workload по умолчанию:

```bash
make build
make -C benchmarks prepare
make -C benchmarks verify
```

Для SPAN:

```bash
make -C benchmarks prepare \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
```

Первый запуск скачивает около 3.7 GB lossless Sintel source. Оба workload
переиспользуют этот source и подготовленные clips. Model weights, generated ONNX
и clips остаются в игнорируемых `models/` и `videos/`.

Только clips можно пересоздать без повторного model export:

```bash
make -C benchmarks prepare ARGS=--force-clips
```

SPAN weights имеют лицензию `CC-BY-NC-SA-4.0`; benchmark tooling не распространяет
веса и сохраняет license/attribution в asset lock.

## Images And Plans

```bash
make -C benchmarks build
make -C benchmarks build-vstrt
make -C benchmarks build-vsgan
```

Проверка command generation не требует GPU. Для VSGAN plan нужен путь будущего
TRT10 engine, но сам файл в dry-run не обязателен:

```bash
make -C benchmarks dry-run \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine \
  VARIANT=720p \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250" \
  VSTRT_ARGS="--warmup-frames 24" \
  VSGAN_ARGS="--warmup-frames 24"
```

Для SPAN вместе с `MANIFEST` переопределяются model paths:

```bash
MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx
ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine

make -C benchmarks dry-run \
  MANIFEST="$MANIFEST" ONNX="$ONNX" ENGINE="$ENGINE" \
  VSGAN_ENGINE="$VSGAN_ENGINE"
```

Параметры frames/runs можно уменьшать только для smoke. Такой suite может быть
валидным, но получает `publishable: false`. Production image не содержит NVML и
внешние benchmark tools. GPU acceptance описан в `GPU_RUNBOOK.md`.
