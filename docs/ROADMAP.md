# Roadmap

Короткий актуальный план.

## Next

- Автоматизировать проверку GPU smoke output: полный decode через
  `ffmpeg -f null -`, FPS, duration, frame count, `has_b_frames`, keyframe
  interval и монотонность PTS/DTS в packet metadata.

## Later

- Улучшить media contract: VFR, rotation, SAR/DAR, duration, missing `nb_frames`.
- Расширить color/bit-depth поддержку за пределы SDR 8-bit NV12/yuv420p:
  P010, HDR metadata passthrough, tonemap/color management.
- Проверить CPU bottleneck в `nvcodec` orchestration на лёгких моделях:
  decoder/encoder queue, async write, PyNvVideoCodec options.
- Исследовать multi-frame async pipeline для `nvcodec`: снять GPU timeline через
  Nsight Systems, проверить простои между NVDEC, CV-CUDA, TensorRT и NVENC; при
  подтверждённых простоях прототипировать ring buffer, несколько CUDA streams и
  event dependencies, затем измерить throughput на тяжёлой и лёгкой моделях.
- Добавить experimental VapourSynth/vs-mlrt/vstrt backend рядом с текущими
  `ffmpeg` и `nvcodec`, не заменяя native GPU path.
- Рассмотреть RIFE/frame interpolation отдельным этапом после video metadata
  и timestamp contract.
