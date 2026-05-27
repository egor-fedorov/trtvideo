# Roadmap

Короткий актуальный план. Подробный исторический план сохранён в
`docs/archive/TASKS.md`.

## Next

- Проверить smoke/batch после color metadata и `--bitrate-mbps` изменений:
  SPAN 720p/1080p через `nvcodec`, затем полный batch.
- Разделить encoder quality API: оставить настоящий `--crf` для `ffmpeg`,
  для `nvcodec` перейти на явные `--bitrate-mbps`, `--max-bitrate-mbps`,
  `--nvenc-rc`, `--constqp`/quality options.
- Обновить benchmark/performance log после новых smoke/batch результатов.

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
