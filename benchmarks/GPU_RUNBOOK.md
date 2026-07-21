# GPU Benchmark Runbook

Этот runbook подготавливает одну benchmark campaign на физической RTX 3090.
Методология и критерии валидности находятся в `methodology.md`. Все команды ниже
выполняются из корня репозитория.

## 1. Build And Assets

```bash
make build
make -C benchmarks build-vstrt
make -C benchmarks build-video2x
make -C benchmarks prepare
make -C benchmarks verify
```

При переходе с workload v1 на `realesrgan-x2plus-sintel-v2` переencode-ьте только
video clips; исходный Y4M и model assets будут переиспользованы:

```bash
make -C benchmarks prepare ARGS=--force-clips
make -C benchmarks verify
```

`prepare` не использует GPU. TensorRT engines, напротив, нужно собирать именно на
той RTX 3090 и в том image, где будет выполняться campaign:

```bash
mkdir -p models/benchmarks/realesrgan-x2plus/engines models/cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_720p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  --timing-cache models/cache/benchmark-rtx3090.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_1080p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  --timing-cache models/cache/benchmark-rtx3090.cache
```

## 2. Offline Plans

До GPU smoke проверьте mounts, paths и точные команды:

```bash
make -C benchmarks dry-run \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  VARIANT=720p \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250" \
  VSTRT_ARGS="--warmup-frames 24" \
  VIDEO2X_ARGS="--warmup-frames 24"
```

## 3. GPU Smoke

Запускайте реализации отдельно. После каждого запуска runner обязан полностью
декодировать output и проверить media/timestamp contract.

```bash
ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine

make -C benchmarks run-ai-media VARIANT=720p ENGINE="$ENGINE" \
  ARGS="--frames 120 --warmup-frames 24 --runs 1 --extra-runs 0 --idle-seconds 0"

make -C benchmarks run-trtexec VARIANT=720p ENGINE="$ENGINE" \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250"

make -C benchmarks run-vstrt VARIANT=720p ENGINE="$ENGINE" \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  VSTRT_ARGS="--warmup-frames 24"

make -C benchmarks run-video2x VARIANT=720p \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  VIDEO2X_ARGS="--warmup-frames 24"
```

Повторите smoke для `VARIANT=1080p` и соответствующего engine. Не переходите к
полным сериям, пока все четыре запуска не завершатся валидным result manifest.

## 4. Full Campaign

Defaults workload manifest задают 100 warmup frames, 1000 measured frames, три
run и ещё два при spread больше 5%. Запуск одной реализации:

```bash
make -C benchmarks run-ai-media VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine
```

Аналогично используются `run-trtexec`, `run-vstrt` и `run-video2x`. До появления
общего campaign runner не запускайте четыре полных suite подряд как финальное
сравнение: методология требует ротировать продукты между раундами. Автоматизация
этой ротации выполняется после GPU acceptance стабильности отдельных runners.

Raw manifests, logs и NVML samples сохраняются в `artefacts/benchmarks/` и не
коммитятся. Публичные sanitized results добавляются только после проверки
environment, hashes и сопоставимости output contract.
