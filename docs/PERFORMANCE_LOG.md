# Performance Log

Historical record of performance changes. See `README.md` for current commands.
For every new performance change, record what changed, the benchmark or command,
the gain or regression, and where it occurred.
Keep entries in reverse chronological order, with the newest entry first.

Entries from before the runtime registry was removed contain historical commands
with `--model` and precision filters. The current CLI requires an explicit
`--engine` instead.

## 2026-07-25 - RTX 3090 multi-resolution single-stream parity baseline

The current single-stream baseline compares `ai-media-enhancer`, TensorRT 11
`vs-mlrt`, and a pinned upstream TensorRT 10.16 VSGAN runtime on
RealESRGAN_x2plus and SPAN at `720p -> 1440p` and `1080p -> 4K`. Both
VapourSynth runners used one vspipe request, one TensorRT stream, and disabled
CUDA Graph; these are parity settings rather than upstream-default or
best-tuned product settings. Measured revision `0fc3037` includes media
preservation. Each campaign used 100 warmup frames, 1000 measured frames, and
three rotated rounds; SPAN 1080p extended to five rounds under the stability
policy. No reduced GPU power limit was applied: the RTX 3090 used its default
350 W board limit.

Key results:

| Workload | Input | ai-media | vs-mlrt | VSGAN | trtexec |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 1080p | 2.884 FPS | 2.394 FPS | 2.399 FPS | 2.921 QPS |
| RealESRGAN_x2plus | 720p | 6.277 FPS | 5.406 FPS | 5.477 FPS | 6.458 QPS |
| SPAN | 1080p | 25.104 FPS | 9.348 FPS | 9.018 FPS | 28.490 QPS |
| SPAN | 720p | 49.941 FPS | 19.825 FPS | 20.315 FPS | 62.846 QPS |

These values remain valid for the recorded single-stream contract. In
particular, the SPAN deltas are not a claim against maximum or upstream-default
VSGAN/vstrt throughput; separate product-default/tuned measurements are
required.

All eight model-space/product-output gates passed. The SPAN 1080p `vs-mlrt`
series is valid with one explicit outlier: the full five-run spread is 6.10%,
while rounds 1, 3, 4, and 5 form a 2.29% consensus. All five runs remain in the
published median.

The same-GPU SPAN 1080p Nsight trace found that CUDA kernel intervals covered
98.17% of the frame loop, with no material per-frame H2D/D2H transfer. NVDEC
and NVENC workloads overlapped CUDA kernels by 91.25% and 98.92%. Profiler FPS
is not a performance result.

The complete compact result, CPU/resource and lifecycle metrics, quality,
hashes, stability evidence, and encoder bitrate caveat are published in the
[RTX 3090 benchmark](../benchmarks/results/rtx-3090/README.md). The ignored raw
artifacts remain outside Git.

## 2026-07-12 - NVENC stream synchronization

Changes:

* The non-profile `nvcodec` path now uses the explicit `runtime.stream` for the
  GPU chain: `NV12->RGB -> TensorRT -> RGB->NV12`.
* `PyNvVideoCodec.CreateEncoder` now receives the same CUDA stream through
  `cudastream=int(runtime.stream.cuda_stream)`.
* The per-frame `stream.synchronize()` before `NVENC Encode` was removed.
  Operation ordering is now provided by the shared CUDA stream.

Benchmark and validation:

* Smoke input: 720p video; output: `3840x2160` at 60 FPS.
* Benchmark inputs: `videos/new_york_720p.mp4`,
  `videos/new_york_1080p.mp4`.
* Benchmark engines:
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_720p_fp16.engine`,
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16.engine`.
* Backend: `nvcodec`.
* GPU: Quadro RTX 6000.
* Benchmark command:
  `benchmark-upscale --engine <resolution-specific-engine> --backend nvcodec --warmup-frames 20 --frames 1000`.
* Smoke workload: complete 120-second output, 7200 frames.
* Benchmark workload: 20 warmup frames and 1000 measured frames.
* Profilers: `py-spy record` and `perf report` for the `upscale` process.
* Validation: `ffmpeg -v error -i <output> -f null -` completed successfully.
* ffprobe output: `width=3840`, `height=2160`, `avg_frame_rate=60/1`,
  `duration=120.000000`, `nb_frames=7200`, `has_b_frames=0`.

Results:

| Metric | Before | After |
| --- | ---: | ---: |
| Dominant CPU stack | `cuStreamSynchronize` / `cudaStreamSynchronize` | `NVDecoder::HandlePictureDisplay` / decoder path |
| `torch.cuda.streams.synchronize` in py-spy | 85.36% after the explicit sync experiment | no longer dominant |
| Process CPU load | one hot core | about 2% on average |
| Output decode validation | passed | passed |

The process-load row is an OS-observed value from that profiling session, not
the later benchmark runner's `getrusage(RUSAGE_CHILDREN)` `average_cores`
metric. It is therefore not directly comparable with current campaign CPU
results. The current pipeline also synchronizes once per decoded batch before
PyNvVideoCodec may reuse its NVDEC surfaces; that correctness synchronization
was added later and was not part of this measurement.

Benchmark after moving NVENC to the runtime CUDA stream:

| Input | Metric | `26.06` baseline | After stream fix | Change |
| --- | ---: | ---: | ---: | ---: |
| 720p | processing FPS | 38.69 | 38.59 | -0.3% |
| 720p | throughput FPS | 37.94 | 37.76 | -0.5% |
| 720p | avg frame time | 25.85 ms | 25.91 ms | +0.3% |
| 720p | `TRT inference` stage | 24.42 ms | 24.36 ms | -0.2% |
| 1080p | processing FPS | 18.12 | 17.98 | -0.8% |
| 1080p | throughput FPS | 17.87 | 17.70 | -1.0% |
| 1080p | avg frame time | 55.17 ms | 55.61 ms | +0.8% |
| 1080p | `TRT inference` stage | 53.59 ms | 53.66 ms | +0.1% |

Conclusion:

* Host-side busy-wait on `cuStreamSynchronize` was eliminated for the regular
  non-profile `nvcodec` run.
* Smoke output correctness was confirmed by full decoding and the basic ffprobe
  contract.
* Throughput did not improve on the heavy SPAN model. Results are within
  single-run noise and are 0.3-1.0% below baseline in some cases.
* The change reduces CPU busy-wait and host-core load; it does not demonstrate an
  FPS improvement for the current SPAN workload.
* The next useful measurement is a lightweight model, where host-side
  synchronization overhead should account for a larger share of each frame.

## 2026-07-12 - TensorRT base image 26.04 -> 26.06

Changes:

* The base image was updated from `nvcr.io/nvidia/tensorrt:26.04-py3` to
  `nvcr.io/nvidia/tensorrt:26.06-py3`.
* The `26.06` engines were rebuilt from mixed-precision FP16 ONNX models after
  migrating to TensorRT 11 strong typing.
* The comparison reflects an update of the entire TensorRT stack and an engine
  rebuild, not the isolated effect of the base Docker image version.

Benchmark:

* Inputs: `videos/new_york_720p.mp4`, `videos/new_york_1080p.mp4`.
* Model: `2xLiveActionV1_SPAN_490000`.
* Backend: `nvcodec`.
* GPU: Quadro RTX 6000.
* Workload: 20 warmup frames and 1000 measured frames.
* Command:
  `benchmark-upscale --engine <resolution-specific-engine> --backend nvcodec --warmup-frames 20 --frames 1000`.
* The CUDA Graph command additionally uses `--cuda-graph`.

Comparison of `26.04` and `26.06` without CUDA Graph:

| Input | Metric | `26.04` | `26.06` | Change |
| --- | ---: | ---: | ---: | ---: |
| 720p | processing FPS | 38.45 | 38.69 | +0.6% |
| 720p | throughput FPS | 37.68 | 37.94 | +0.7% |
| 720p | avg frame time | 26.01 ms | 25.85 ms | -0.6% |
| 720p | `TRT inference` stage | 24.41 ms | 24.42 ms | +0.0% |
| 1080p | processing FPS | 17.90 | 18.12 | +1.2% |
| 1080p | throughput FPS | 17.66 | 17.87 | +1.2% |
| 1080p | avg frame time | 55.86 ms | 55.17 ms | -1.2% |
| 1080p | `TRT inference` stage | 54.09 ms | 53.59 ms | -0.9% |

CUDA Graph effect within `26.06`:

| Input | Metric | Regular enqueue | CUDA Graph | Change |
| --- | ---: | ---: | ---: | ---: |
| 720p | processing FPS | 38.69 | 39.13 | +1.1% |
| 720p | throughput FPS | 37.94 | 38.29 | +0.9% |
| 720p | avg frame time | 25.85 ms | 25.55 ms | -1.1% |
| 720p | `TRT inference` stage | 24.42 ms | 24.35 ms | -0.3% |
| 1080p | processing FPS | 18.12 | 18.05 | -0.4% |
| 1080p | throughput FPS | 17.87 | 17.79 | -0.4% |
| 1080p | avg frame time | 55.17 ms | 55.39 ms | +0.4% |
| 1080p | `TRT inference` stage | 53.59 ms | 53.85 ms | +0.5% |

Conclusion:

* The `26.04` to `26.06` migration caused no performance regression on these
  SPAN workloads; results improved by 0.6-1.2%.
* The difference is close to typical single-run noise, so a stable speedup cannot
  be claimed without repeated runs.
* CUDA Graph still provides no consistent gain for the heavy SPAN model: a small
  improvement at 720p becomes a small regression at 1080p.
* Keep `--cuda-graph` as an experimental opt-in setting rather than enabling it
  by default.

## 2026-05-12 - Docker base image refresh

Changes:

* The base image was updated from `nvcr.io/nvidia/tensorrt:26.03-py3` to
  `nvcr.io/nvidia/tensorrt:26.04-py3`.
* The pipeline code did not change.
* The `26.04` benchmark used an engine rebuilt in the new image. This therefore
  compares a complete TensorRT stack refresh, not a pure runtime-only A/B test
  using the same `.engine`.

Benchmark:

* Inputs: `videos/switzerland_720p.mp4`,
  `videos/switzerland_1080p.mp4`.
* Engine: FP16 I/O engines from `models/liveaction-span/engines/`; the `26.04`
  engine was rebuilt with the new TensorRT runtime.
* GPU: Quadro RTX 6000.
* Command:
  `benchmark-upscale --model models/liveaction-span --backend nvcodec --engine-io-precision fp16 --warmup-frames 20 --frames 1000`.
* The CUDA Graph command adds `--cuda-graph`.
* For 1080p without CUDA Graph, the `26.04` column is the average of two runs.

Comparison of the `26.03` and `26.04` stacks without CUDA Graph:

| Input | Metric | `26.03` FP16 I/O | `26.04` FP16 I/O | Change |
| --- | ---: | ---: | ---: | ---: |
| 720p | processing FPS | 37.76 | 37.22 | -1.4% |
| 720p | throughput FPS | 36.75 | 35.81 | -2.5% |
| 720p | avg frame time | 26.49 ms | 26.87 ms | +1.4% |
| 720p | `TRT inference` stage | 25.29 ms | 25.00 ms | -1.1% |
| 1080p | processing FPS | 17.51 | 16.78 | -4.2% |
| 1080p | throughput FPS | 17.20 | 15.93 | -7.4% |
| 1080p | avg frame time | 57.10 ms | 59.58 ms | +4.3% |
| 1080p | `TRT inference` stage | 55.71 ms | 57.64 ms | +3.5% |

CUDA Graph on `26.04`:

| Input | Metric | FP16 I/O | FP16 I/O + CUDA Graph | Change |
| --- | ---: | ---: | ---: | ---: |
| 720p | `cuda_graph` | false | true | capture works |
| 720p | processing FPS | 37.22 | 36.99 | -0.6% |
| 720p | throughput FPS | 35.81 | 34.09 | -4.8% |
| 720p | avg frame time | 26.87 ms | 27.04 ms | +0.6% |
| 720p | `TRT inference` stage | 25.00 ms | 24.97 ms | -0.1% |
| 1080p | `cuda_graph` | false | true | capture works |
| 1080p | processing FPS | 16.78 | 16.14 | -3.8% |
| 1080p | throughput FPS | 15.93 | 15.72 | -1.3% |
| 1080p | avg frame time | 59.58 ms | 61.94 ms | +4.0% |
| 1080p | `TRT inference` stage | 57.64 ms | 59.95 ms | +4.0% |

Conclusion:

* Updating to `26.04` and rebuilding the engine did not improve performance for
  the current SPAN FP16 I/O model on the Quadro RTX 6000.
* 720p is nearly within noise, but 1080p became slower in both processing FPS and
  the TensorRT stage.
* CUDA Graph capture works on `26.04`, but provides no stable gain for 720p or
  1080p SPAN workloads.
* Do not enable `--cuda-graph` by default; keep it experimental.
* `build-engine` in the new image was validated by rebuilding the engine.
* A pure runtime-only A/B test would require running the old `26.03` engine in
  the `26.04` image, but that result may not be representative because of
  TensorRT engine/runtime compatibility.

## 2026-05-11 - NVDEC/NVENC buffer pool

Changes:

* Added a per-job `FrameBufferPool` in `ai_media/pipelines/nvcodec.py`.
* The `NV12->RGB` and `RGB->NV12` CV-CUDA conversions now write into preallocated
  buffers.
* `TensorRTRuntime.infer_rgb_tensor_into(...)` writes output into a preallocated
  RGB buffer.
* The `upscale --backend nvcodec` hot path reuses `nv12_in`, `rgb_in`, `nchw_in`,
  `rgb_out`, `rgb_out_float`, and `nv12_out`.

Benchmark:

* Input: `videos/switzerland_720p.mp4`.
* Engine:
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_720p.engine`.
* GPU: Quadro RTX 6000.
* Command:
  `benchmark-upscale --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`.

Results:

| Backend | Metric | Before | After | Change |
| --- | ---: | ---: | ---: | ---: |
| `nvcodec` | processing FPS | 36.5 | 37.44 | +2.6% |
| `nvcodec` | avg frame time | 27.4 ms | 26.71 ms | -2.5% |
| `nvcodec` | `NV12->RGB` stage | 0.8 ms | 0.50 ms | -37.5% |
| `nvcodec` | `TRT inference` stage | 25.7 ms | 25.49 ms | -0.8% |

Conclusion:

* The improvement is visible in the `NV12->RGB` CV-CUDA stage.
* Overall FPS improved moderately because TensorRT inference still dominates the
  720p SPAN run.
* The benchmark also measures the `ffmpeg` backend, but this optimization did
  not target `ffmpeg`.
* Use `throughput_fps` for end-to-end performance and `processing_fps` plus
  `stage_ms` for hot-path analysis.

## 2026-05-11 - FP16 I/O experiment

Changes:

* Added experimental `build-engine --fp16-io`.
* TensorRT input and output bindings can be built as FP16 instead of FP32.
* Registry selection supports `--engine-io-precision fp16|fp32`.
* The runtime allocates input and output buffers according to engine binding
  dtypes.
* The preprocess and postprocess paths support FP16 bindings in
  `upscale --backend ffmpeg` and `upscale --backend nvcodec`.

Benchmark:

* Input: `videos/switzerland_1080p.mp4`.
* Default engine:
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p.engine`.
* FP16 I/O engine:
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`.
* GPU: Quadro RTX 6000.
* Command:
  `benchmark-upscale --model models/liveaction-span --backend ffmpeg,nvcodec --warmup-frames 20 --frames 1000`.
* The FP16 I/O command adds `--engine-io-precision fp16`.

Results:

| Backend | Metric | Default FP16 / FP32 I/O | FP16 I/O | Change |
| --- | ---: | ---: | ---: | ---: |
| `ffmpeg` | throughput FPS | 7.61 | 7.84 | +3.0% |
| `ffmpeg` | processing FPS | 16.27 | 16.66 | +2.4% |
| `ffmpeg` | avg frame time | 61.48 ms | 60.02 ms | -2.4% |
| `ffmpeg` | GPU peak memory | 261.84 MB | 143.87 MB | -45.1% |
| `ffmpeg` | postprocess stage | 1.60 ms | 0.94 ms | -41.3% |
| `ffmpeg` | TRT stage | 52.80 ms | 52.07 ms | -1.4% |
| `nvcodec` | throughput FPS | 16.53 | 17.20 | +4.1% |
| `nvcodec` | processing FPS | 16.80 | 17.51 | +4.2% |
| `nvcodec` | avg frame time | 59.51 ms | 57.10 ms | -4.1% |
| `nvcodec` | GPU peak memory | 282.74 MB | 164.90 MB | -41.7% |
| `nvcodec` | `TRT inference` stage | 58.19 ms | 55.71 ms | -4.3% |

Conclusion:

* FP16 I/O provides a moderate speedup for 1080p to 4K, especially with
  `nvcodec`.
* The main effect is a peak GPU memory reduction of approximately 42-45%.
* TensorRT compute still dominates 1080p SPAN inference.
* Before making this the production default, separately validate visual
  artifacts, banding or clipping, and compatibility with other models.

## 2026-05-11 - CUDA Graph experiment

Changes:

* Added experimental `--cuda-graph`.
* The benchmark harness passes `--cuda-graph` to
  `upscale --backend ffmpeg|nvcodec`.
* `TensorRTRuntime` attempts to capture TensorRT `execute_async_v3` in a CUDA
  Graph.
* If capture fails, the runtime falls back to regular TensorRT enqueue.

Benchmark:

* Input: `videos/switzerland_1080p.mp4`.
* Engine:
  `models/liveaction-span/engines/2xLiveActionV1_SPAN_490000_1080p_fp16io.engine`.
* GPU: Quadro RTX 6000.
* Command:
  `benchmark-upscale --model models/liveaction-span --backend nvcodec --engine-io-precision fp16 --cuda-graph --warmup-frames 20 --frames 1000`.

Results:

| Backend | Metric | FP16 I/O | FP16 I/O + CUDA Graph | Change |
| --- | ---: | ---: | ---: | ---: |
| `nvcodec` | `cuda_graph` | false | true | capture works |
| `nvcodec` | processing FPS | 17.51 | 17.86 | +2.0% |
| `nvcodec` | throughput FPS | 17.20 | 17.15 | -0.3% |
| `nvcodec` | avg frame time | 57.10 ms | 55.98 ms | -2.0% |
| `nvcodec` | `TRT inference` stage | 55.71 ms | 54.47 ms | -2.2% |
| `nvcodec` | GPU peak memory | 164.90 MB | 164.90 MB | unchanged |

Conclusion:

* CUDA Graph capture is now active: `cuda_graph: true`,
  `cuda_graph_error: null`.
* The effect is small on the heavy 1080p SPAN model because frame time is
  dominated by TensorRT compute.
* End-to-end throughput is effectively within noise, so the production default
  should not change yet.
* The next useful measurement is a lightweight or compact model, where CPU
  launch overhead should be more visible.

## 2026-05-11 - Docker dependency layer/cache

Changes:

* Docker runtime dependencies are read from `pyproject.toml` and `uv.lock`
  through `uv export --frozen --no-emit-project`.
* Dev-only dependencies are defined in `[dependency-groups].dev` and installed
  with `--group dev` when `--build-arg INSTALL_DEV=1` is set.
* Before application code is copied, the Dockerfile installs dependencies into
  the `/opt/ai-media-enhancer` virtual environment with
  `--system-site-packages`.
* The virtual environment can access packages preinstalled in the TensorRT base
  image without modifying managed `/usr`.
* The uv download and wheel cache uses a BuildKit cache mount.
* Application code is installed separately with the fast
  `uv pip install --python "$VIRTUAL_ENV" --no-deps .` command.
* `.dockerignore` excludes local cache directories, benchmark JSON artifacts,
  and temporary logs.

Confirmed effect:

* A warm-cache rebuild after a code-only change reuses the dependency layer.
* Heavy dependencies are no longer reinstalled: `torch`, `cvcuda`,
  `pynvvideocodec`, `onnx`, `onnxscript`, and `spandrel`.
* The production image should not contain the uv or pip download cache in its
  final layer.
* The exact time saved depends on the host cache and Docker storage driver, so
  record exact seconds only together with the complete build log.
