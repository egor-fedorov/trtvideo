# Runbook: RealESRGAN export/build и прогон samples через две модели

Инструкция рассчитана на запуск из корня репозитория через Docker.

Что делает:

* собирает актуальный Docker image;
* экспортирует RealESRGAN `.pth` в static ONNX для 720p и 1080p;
* собирает TensorRT FP16 I/O engines для RealESRGAN и обновляет registry;
* собирает TensorRT FP16 I/O engines для SPAN и обновляет registry;
* прогоняет файлы из `../samples`, в имени которых есть `720` или `1080`, через две модели:
  `models/liveaction-span` и `models/realesrgan-x2plus`.

## 0. Переменные

Если `.pth` лежит в другом месте, поменяйте только `REALESRGAN_PTH`.

```bash
export IMAGE=ai-media-enhancer:latest
export SAMPLES_DIR="$PWD/../samples"
export OUTPUT_DIR="$PWD/artefacts/sample-runs"
export REALESRGAN_PTH=models/pth/RealESRGAN_x2plus.pth
export REALESRGAN_MODEL=models/realesrgan-x2plus
export SPAN_MODEL=models/liveaction-span
export SPAN_ONNX_720=models/onnx/2xLiveActionV1_SPAN_490000_720p.onnx
export SPAN_ONNX_1080=models/onnx/2xLiveActionV1_SPAN_490000_1080p.onnx
export BACKEND=nvcodec
export ENGINE_IO_PRECISION=fp16
export NVENC_BITRATE_MBPS=45
```

## 1. Собрать Docker image

```bash
DOCKER_BUILDKIT=1 docker build -t "$IMAGE" .
```

Проверить CLI:

```bash
docker run --rm "$IMAGE" upscale --help
docker run --rm "$IMAGE" export-onnx --help
docker run --rm "$IMAGE" build-engine --help
```

## 2. RealESRGAN `.pth` -> ONNX

Команда создаёт:

```text
models/onnx/realesrgan_x2plus_720p.onnx
models/onnx/realesrgan_x2plus_1080p.onnx
```

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  "$IMAGE" export-onnx \
  --model_path "$REALESRGAN_PTH"
```

## 3. Собрать SPAN TensorRT engines

Если static ONNX для SPAN ещё не создан, сначала подготовьте его через `prepare-onnx`
или укажите другие пути в `SPAN_ONNX_720` и `SPAN_ONNX_1080`.

Создать директории:

```bash
mkdir -p "$PWD/$SPAN_MODEL/engines" "$PWD/models/cache"
```

720p engine:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  "$IMAGE" build-engine \
  "$SPAN_ONNX_720" \
  -o "$SPAN_MODEL/engines/2xLiveActionV1_SPAN_490000_720p_fp16io.engine" \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry "$SPAN_MODEL"
```

1080p engine:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  "$IMAGE" build-engine \
  "$SPAN_ONNX_1080" \
  -o "$SPAN_MODEL/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine" \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry "$SPAN_MODEL"
```

Проверить registry:

```bash
find "$PWD/$SPAN_MODEL" -maxdepth 3 -type f | sort
```

## 4. Собрать RealESRGAN TensorRT engines

Создать директории:

```bash
mkdir -p "$PWD/$REALESRGAN_MODEL/engines" "$PWD/models/cache"
```

720p engine:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  "$IMAGE" build-engine \
  models/onnx/realesrgan_x2plus_720p.onnx \
  -o "$REALESRGAN_MODEL/engines/realesrgan_x2plus_720p_fp16io.engine" \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry "$REALESRGAN_MODEL"
```

1080p engine:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  "$IMAGE" build-engine \
  models/onnx/realesrgan_x2plus_1080p.onnx \
  -o "$REALESRGAN_MODEL/engines/realesrgan_x2plus_1080p_fp16io.engine" \
  --fp16-io \
  --timing-cache models/cache/trt.cache \
  --registry "$REALESRGAN_MODEL"
```

Проверить registry:

```bash
find "$PWD/$REALESRGAN_MODEL" -maxdepth 3 -type f | sort
```

## 5. Проверить registries

```bash
find "$PWD/$SPAN_MODEL" -maxdepth 3 -type f | sort
find "$PWD/$REALESRGAN_MODEL" -maxdepth 3 -type f | sort
```

Если вы собрали engines без `--fp16-io`, перед batch-прогоном установите:

```bash
export ENGINE_IO_PRECISION=fp32
```

## 6. Прогнать samples через обе модели

Output layout:

```text
artefacts/sample-runs/
  liveaction-span/
  realesrgan-x2plus/
```

Batch command. `upscale` печатает полный лог каждого запуска в stdout, а в конце
команда дополнительно выводит компактную FPS-таблицу по всем обработанным файлам.
Для итоговой таблицы используется `Without warmup` FPS из stdout каждого запуска.

```bash
mkdir -p "$OUTPUT_DIR/liveaction-span" "$OUTPUT_DIR/realesrgan-x2plus"

summary_rows=""
exec 3>&1

for model in "$SPAN_MODEL" "$REALESRGAN_MODEL"; do
  model_name="$(basename "$model")"

  while IFS= read -r -d '' sample; do
    name="$(basename "$sample")"
    stem="${name%.*}"
    out_file="out/$model_name/${stem}_${model_name}_${BACKEND}.mp4"
    bitrate_args=()
    if [ "$BACKEND" = "nvcodec" ] && [ -n "${NVENC_BITRATE_MBPS:-}" ]; then
      bitrate_args=(--bitrate-mbps "$NVENC_BITRATE_MBPS")
    fi

    echo "=== $name -> $model_name ($BACKEND) ==="
    run_status=0
    run_output="$(
      set -o pipefail
      docker run --rm --gpus all \
        -v "$PWD/models:/app/models" \
        -v "$SAMPLES_DIR:/app/samples:ro" \
        -v "$OUTPUT_DIR:/app/out" \
        "$IMAGE" upscale \
        --backend "$BACKEND" \
        --model "$model" \
        --engine-io-precision "$ENGINE_IO_PRECISION" \
        "${bitrate_args[@]}" \
        --input "samples/$name" \
        --output "$out_file" \
        --log-interval 100 2>&1 | tee /dev/fd/3
    )" || run_status=$?

    if [ "$run_status" -ne 0 ]; then
      exec 3>&-
      exit "$run_status"
    fi

    fps="$(printf '%s\n' "$run_output" |
      sed -nE 's/.*Without warmup:.*\(([0-9.]+) fps\).*/\1/p' |
      tail -n 1)"
    if [ -z "$fps" ]; then
      fps="n/a"
    fi

    printf -v row '%-34s %-20s %-8s %10s\n' "$name" "$model_name" "$BACKEND" "$fps"
    summary_rows+="$row"
  done < <(find "$SAMPLES_DIR" -maxdepth 1 -type f \( -name '*720*.mp4' -o -name '*1080*.mp4' \) -print0)
done

exec 3>&-

printf '\n=== FPS summary, without warmup ===\n'
printf '%-34s %-20s %-8s %10s\n' "input" "model" "backend" "fps"
printf '%s' "$summary_rows"
```

## 7. Быстрый smoke на одном файле

Перед полным batch можно проверить один `720` sample:

```bash
sample="$(find "$SAMPLES_DIR" -maxdepth 1 -type f -name '*720*.mp4' | head -n 1)"
name="$(basename "$sample")"

docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$SAMPLES_DIR:/app/samples:ro" \
  -v "$OUTPUT_DIR:/app/out" \
  "$IMAGE" upscale \
  --backend "$BACKEND" \
  --model "$REALESRGAN_MODEL" \
  --engine-io-precision "$ENGINE_IO_PRECISION" \
  --bitrate-mbps "$NVENC_BITRATE_MBPS" \
  --input "samples/$name" \
  --output "out/${name%.*}_realesrgan_smoke.mp4" \
  --max-frames 30 \
  --log-interval 10
```
