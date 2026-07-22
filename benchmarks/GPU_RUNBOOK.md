# GPU Benchmark Runbook

Runbook предназначен для acceptance runner’ов на одной физической RTX 3090.
Rotated campaign и exact rate control реализованы, но публикационный результат
по-прежнему требует CPU/timing scopes и quality parity из `methodology.md`. Все
команды выполняются из корня репозитория.

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

`build-vsgan` скачивает pinned full `latest_no_avx512` image размером около
13 GB. Он выбран вместо сломанного `minimal_no_avx512`, в котором отсутствует
рабочий нативный `vspipe`. Wrapper устанавливает pinned Ubuntu FFmpeg 6.1.1 для
совместимого NVENC encode и output validation: upstream FFmpeg требует driver
610+.

## 2. TensorRT Engines

TRT11 engines проекта собираются production image на benchmark GPU. Соберите
варианты 720p и 1080p для обеих моделей:

```bash
mkdir -p \
  models/benchmarks/realesrgan-x2plus/engines \
  models/benchmarks/liveaction-span/engines \
  models/cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_720p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_720p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_1080p_fp16.onnx \
  -o models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/liveaction-span/onnx/liveaction_span_720p_fp16.onnx \
  -o models/benchmarks/liveaction-span/engines/liveaction_span_720p.engine \
  --timing-cache models/cache/benchmark-trt11.cache

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  ai-media-enhancer:latest build-engine \
  models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx \
  -o models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  --timing-cache models/cache/benchmark-trt11.cache
```

Stock VSGAN использует TensorRT 10.16, поэтому получает отдельный engine из того
же ONNX. Builder сохраняет log и sidecar:

```bash
make -C benchmarks build-vsgan-engine VARIANT=720p
make -C benchmarks build-vsgan-engine VARIANT=1080p

make -C benchmarks build-vsgan-engine \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=720p \
  ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_720p_fp16.onnx \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_720p.engine

make -C benchmarks build-vsgan-engine \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

VSGAN engines необходимо пересобирать после изменения pinned VSGAN base image:
TensorRT serialized plans не совместимы между разными runtime builds. Runner
проверяет сохранённый base-image digest до начала warmup.

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

## 5. Rotated Acceptance Campaign

Перед campaign закоммитьте изменения и заново соберите все три benchmark image.
Preflight отклоняет dirty worktree и image, собранные не из текущего commit.

Canonical defaults: 100 warmup frames, 1000 measured frames, три чередующихся
раунда и ещё два при spread хотя бы одной реализации больше 5%:

```bash
make -C benchmarks run-campaign \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

Для SPAN:

```bash
make -C benchmarks run-campaign \
  CAMPAIGN_NAME=liveaction-span-sintel-1080p \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json \
  VARIANT=1080p \
  ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine
```

Повторите обе команды с 720p paths. После безопасного прерывания продолжить ту
же campaign можно с `RESUME=1`. Resume допустим только при неизменных commit,
images, workload assets и engines; partial/invalid round сохраняется для
диагностики и требует ручного удаления только своей директории.

Campaign сохраняет raw manifests и общие `campaign.json`/`results.md` в
`artefacts/benchmarks/campaigns/<name>/`. Пока CPU/timing/quality gates не
закрыты, агрегатор выставляет `publishable: false` даже для валидной campaign.

Individual `run-ai-media`, `run-vstrt` и `run-vsgan` остаются для smoke и
диагностики. `run-trtexec` остаётся отдельным inference ceiling.

Не публикуйте последовательный запуск независимых suite как финальное сравнение.
Raw manifests, logs и NVML samples не коммитятся до sanitization/review.
