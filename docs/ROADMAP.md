# Roadmap

Короткий актуальный план. Подробный исторический план сохранён в
`docs/archive/TASKS.md`.

## Next

- Добавить media validation checks для будущих smoke/integration тестов:
  `ffmpeg -f null -`, `ffprobe has_b_frames`, FPS/duration/frame count и
  монотонность PTS/DTS в packet metadata.

## Later

- Улучшить media contract: VFR, rotation, SAR/DAR, duration, missing `nb_frames`.
- Расширить color/bit-depth поддержку за пределы SDR 8-bit NV12/yuv420p:
  P010, HDR metadata passthrough, tonemap/color management.
- Проверить CPU bottleneck в `nvcodec` orchestration на лёгких моделях:
  decoder/encoder queue, async write, PyNvVideoCodec options.
- Добавить experimental VapourSynth/vs-mlrt/vstrt backend рядом с текущими
  `ffmpeg` и `nvcodec`, не заменяя native GPU path.
- Рассмотреть RIFE/frame interpolation отдельным этапом после video metadata
  и timestamp contract.
