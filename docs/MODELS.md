# Model Sources

Веса моделей, ONNX-файлы и TensorRT engines не хранятся в Git. Репозиторий
работает по схеме bring-your-own-weights: пользователь скачивает веса из upstream,
проверяет лицензию и кладёт файлы в `models/`.

Краткий машинно-читаемый каталог источников хранится в `model_sources.json`. Он не
интегрирован с runtime и нужен только как справочник provenance:
идентификатор модели, upstream URL, download URL и license.

## RealESRGAN_x2plus

- Source release: <https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1>
- Weight: <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth>
- Paper: <https://arxiv.org/abs/2107.10833>
- License: `BSD-3-Clause` для upstream repository; перед redistribution model
  artifact terms нужно перепроверять upstream.
- Expected path: `models/pretrained/RealESRGAN_x2plus.pth`

Release URL intentionally pinned to `v0.2.1`. Это не ссылка на latest release:
обновление upstream не должно незаметно менять используемый artifact.

Базовый workflow:

```bash
mkdir -p models/pretrained
curl -L \
  -o models/pretrained/RealESRGAN_x2plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

После этого используйте `export-onnx` workflow из README.

## 2xLiveActionV1_SPAN

- OpenModelDB page: <https://openmodeldb.info/models/2x-LiveActionV1-SPAN>
- Training/config source: <https://github.com/jcj83429/upscaling/tree/9332e7d5b07747ff347e5abdc43f8144364de9f7/2xLiveActionV1_SPAN>
- Weight: <https://raw.githubusercontent.com/jcj83429/upscaling/f73a3a41106f9c9aa6257556d358740a91b2ddb5/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth>
- ONNX: <https://raw.githubusercontent.com/jcj83429/upscaling/9332e7d5b07747ff347e5abdc43f8144364de9f7/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.onnx>
- License: `CC-BY-NC-SA-4.0`; это non-commercial license, credit required,
  share-alike terms apply.
- Expected dynamic ONNX path: `models/onnx/2xLiveActionV1_SPAN_490000.onnx`

Базовый workflow через готовый ONNX:

```bash
mkdir -p models/onnx
curl -L \
  -o models/onnx/2xLiveActionV1_SPAN_490000.onnx \
  https://raw.githubusercontent.com/jcj83429/upscaling/9332e7d5b07747ff347e5abdc43f8144364de9f7/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.onnx
```

После этого создайте static variants через `prepare-onnx` и соберите TensorRT
engines через `build-engine`.

## Связь С Runtime

`model_sources.json` отвечает только на вопрос "откуда взять исходную модель".
Runtime registry отсутствует: конкретный TensorRT engine пользователь передаёт в
`upscale` или `benchmark-upscale` через `--engine`. Sidecar `<engine>.json` хранит
метаданные сборки конкретного engine, но не участвует в его выборе.

## Проверка Целостности

Если файл скачан вручную, зафиксируйте локальный hash перед benchmark/release
артефактами:

```bash
sha256sum models/pretrained/RealESRGAN_x2plus.pth
sha256sum models/onnx/2xLiveActionV1_SPAN_490000.onnx
```

Поля `sha256` пока не закреплены в `model_sources.json`, потому что сейчас это
только справочник источников, а не исполняемый контракт.
