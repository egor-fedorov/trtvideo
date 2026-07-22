# GPU Benchmark Runbook

Runbook предназначен для acceptance runner’ов на одной физической RTX 3090.
Финальная публикационная campaign требует закрыть measurement gaps из
`methodology.md`: exact rate control, CPU/timing scopes, quality parity и
чередование продуктов. Все команды выполняются из корня репозитория.

## 1. Build And Assets

```bash
make build
make -C benchmarks build
make -C benchmarks build-vstrt
make -C benchmarks build-vsgan

make -C benchmarks prepare
make -C benchmarks verify

make -C benchmarks prepare \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
```

`prepare` не использует GPU. RealESRGAN и SPAN переиспользуют один Sintel source
и clips, но имеют отдельные ONNX directories.

## 2. TensorRT Engines

TRT11 engines проекта собираются production image на benchmark GPU. Пример для
RealESRGAN 1080p:

```bash
mkdir -p models/benchmarks/realesrgan-x2plus/engines models/cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_1080p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  --timing-cache models/cache/benchmark-trt11.cache
```

Повторите для 720p и для ONNX/engine paths из SPAN manifest.

Stock VSGAN использует TensorRT 10.16, поэтому получает отдельный engine из того
же ONNX. Builder сохраняет log и sidecar:

```bash
make -C benchmarks build-vsgan-engine VARIANT=1080p

make -C benchmarks build-vsgan-engine \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

Не копируйте serialized engine между TRT10 и TRT11. Оба engine должны быть
собраны на той же benchmark GPU.

## 3. Offline Plans

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

Проверьте generated commands, mounted paths и pinned implementation metadata.

## 4. GPU Smoke

Зафиксируйте power limit, driver и отсутствие посторонней GPU-нагрузки. Затем
запустите каждый runner отдельно:

```bash
ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine
VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_720p.engine
SMOKE="--frames 120 --warmup-frames 24 --runs 1 --extra-runs 0 --idle-seconds 0"

make -C benchmarks run-ai-media VARIANT=720p ENGINE="$ENGINE" ARGS="$SMOKE"
make -C benchmarks run-vstrt VARIANT=720p ENGINE="$ENGINE" ARGS="$SMOKE"
make -C benchmarks run-vsgan VARIANT=720p \
  VSGAN_ENGINE="$VSGAN_ENGINE" ARGS="$SMOKE"
make -C benchmarks run-trtexec VARIANT=720p ENGINE="$ENGINE" \
  ARGS="--frames 120 --runs 1 --extra-runs 0 --idle-seconds 0" \
  TRTEXEC_ARGS="--warmup-ms 250"
```

Повторите для 1080p, затем для SPAN с переопределёнными `MANIFEST`, `ENGINE`,
`ONNX` и `VSGAN_ENGINE`. Каждый video runner должен полностью декодировать output
и проверить media/timestamp contract. `trtexec` проверяется отдельно как
diagnostic ceiling.

## 5. Acceptance Results

Canonical defaults: 100 warmup frames, 1000 measured frames, три run и ещё два
при spread больше 5%. Individual suite можно снять так:

```bash
make -C benchmarks run-ai-media VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine
```

Аналогично доступны `run-vstrt`, `run-vsgan` и diagnostic `run-trtexec`.

Не публикуйте последовательный запуск этих независимых suite как финальное
сравнение: canonical campaign должна чередовать реализации по раундам и включать
недостающие CPU/timing/quality checks. Raw manifests, logs и NVML samples
сохраняются в `artefacts/benchmarks/` и не коммитятся до sanitization/review.
