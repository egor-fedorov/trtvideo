# Benchmarks

This directory contains reproducible workload manifests, pinned implementation
metadata, isolated Docker environments, and runners. Models, ONNX files,
TensorRT engines, source videos, and raw results are not added to Git.

- `methodology.md` - comparison classes and validity criteria.
- `workloads/` - RealESRGAN and SPAN workload manifests.
- `implementations.json` - pinned diagnostic, parity, and product
  implementations.
- `docker/` - TensorRT 11 vstrt parity and stock VSGAN environments.
- `scripts/` - asset preparation, engine builders, and runners.
- `GPU_RUNBOOK.md` - acceptance sequence on the benchmark GPU.

The benchmark workflow is separated from the root `Makefile`:

```bash
make -C benchmarks help
```

## Matrix

- `run-vstrt` - technical parity with the same TensorRT 11 engine.
- `run-vsgan` - stock product comparison from the same ONNX but with a separate
  TRT10.16 engine because serialized engines are incompatible across runtime
  versions.
- `run-trtexec` - diagnostic inference ceiling, not a competitor.
- `run-campaign` - canonical rotation of project/vstrt/VSGAN by round and
  generation of a shared acceptance table.

Video2X is excluded because it did not run the canonical
`RealESRGAN_x2plus`; its FPS therefore did not answer the same-model performance
question.

## Assets

RealESRGAN is the default workload:

```bash
make build
make -C benchmarks prepare
make -C benchmarks verify
```

For SPAN:

```bash
make -C benchmarks prepare \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
make -C benchmarks verify \
  MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
```

The first run downloads approximately 3.7 GB of lossless Sintel source data.
Both workloads reuse this source and the prepared clips. Model weights,
generated ONNX files, and clips remain in ignored `models/` and `videos/`
directories.

Recreate only the clips without exporting the models again:

```bash
make -C benchmarks prepare ARGS=--force-clips
```

SPAN weights use the `CC-BY-NC-SA-4.0` license. The benchmark tooling does not
redistribute them and records license/attribution data in the asset lock.

## Images And Plans

```bash
make -C benchmarks build
make -C benchmarks build-vstrt
make -C benchmarks build-vsgan
```

The VSGAN wrapper uses the pinned `latest_no_avx512` image. In the corresponding
`minimal_no_avx512` release, the native `vspipe` binary was replaced by an
incompatible Python entrypoint. The benchmark runner starts from a separate
virtual environment by absolute path without activating it for the embedded
Python inside VSScript. The Docker build validates both Python environments and
the native binary.

Upstream FFmpeg requires NVENC API 13.1 and driver 610+. The benchmark wrapper
uses pinned Ubuntu FFmpeg `7:6.1.1-3ubuntu5` as an external encoder adapter and
the source of `ffprobe`; the stock VSGAN inference stack remains unchanged.

Command-generation checks do not require a GPU. A VSGAN plan needs the path of
the future TRT10 engine, but the file itself is optional in dry-run mode:

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

For SPAN, override the model paths together with `MANIFEST`:

```bash
MANIFEST=benchmarks/workloads/liveaction_span_sintel.json
ONNX=models/benchmarks/liveaction-span/onnx/liveaction_span_1080p_fp16.onnx
ENGINE=models/benchmarks/liveaction-span/engines/liveaction_span_1080p.engine
VSGAN_ENGINE=models/benchmarks/liveaction-span/engines/vsgan/liveaction_span_1080p.engine

make -C benchmarks dry-run \
  MANIFEST="$MANIFEST" ONNX="$ONNX" ENGINE="$ENGINE" \
  VSGAN_ENGINE="$VSGAN_ENGINE"
```

Frame/run parameters may be reduced only for smoke tests. Such a suite can be
valid, but it receives `scope: acceptance` and `publishable: false`. The same
restriction applies to a canonical individual suite: comparisons may be
published only from the shared rotated campaign.

All video runners use one explicit NVENC contract: H.264 P4/HQ, CBR,
target=min=max bitrate, a two-second VBV buffer with 50% initial occupancy,
single pass, lookahead/AQ disabled, a one-second GOP, and zero B-frames.

Run the canonical campaign after smoke tests:

```bash
make -C benchmarks run-campaign \
  VARIANT=1080p \
  ENGINE=models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_1080p.engine \
  VSGAN_ENGINE=models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_1080p.engine
```

Results are written to
`artefacts/benchmarks/campaigns/realesrgan_x2plus_sintel-1080p/`: raw manifests,
`campaign.events.jsonl`, `campaign.json`, and `results.md`. The event log records
the actual order, start/end time, and observed pause for each run; the aggregator
rejects results without a complete log. The main table contains median FPS, wall
time, CPU cores, GPU utilization, power, VRAM, bitrate, and size. A separate
lifecycle table contains median startup, steady-state frame loop, and
finalize/mux durations, which sum to the same full-process wall time.

The production image contains neither NVML nor external benchmark tools. The
complete GPU workflow is documented in `GPU_RUNBOOK.md`.
