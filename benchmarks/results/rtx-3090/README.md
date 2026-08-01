# RTX 3090 Comparative Benchmark

This directory contains the current privacy-reviewed RTX 3090 benchmark
evidence. The headline tuned matrix was measured on 2026-07-31 from clean
revision `cb5e645eb4ddcac35c59122bb62fd70cc7e50dcf`. The retained
upstream-default and diagnostic snapshots were measured on 2026-07-30 from
revision `8adbca96b829bd2f791fe5bce5c27029e283b79d`.

Each result class used an RTX 3090, Ryzen 5 5600, driver 595.84, the same model
and clip assets, the same output contract, and an active 350 W board limit. No
reduced power cap was applied. Result classes are independent sessions and are
not compared row-by-row across revisions or physical GPU instances. Every
within-class product comparison used one physical GPU and one clean revision.

The tuned final campaigns used 1000 measured frames, 30 RealESRGAN or 100 SPAN
warmup frames, three rotated rounds, and ten seconds idle between processes.
Model-space and 1000-frame product-output quality gates passed at both
resolutions. Both cross-resolution publication matrices report `valid` and
`publishable`.

## Best-Tuned Results

Tuned candidates were selected by a separate sweep. Selected profiles then
passed fresh quality gates and independent rotated campaigns; sweep FPS is not
used as the final comparison value.

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.078 FPS | 6.151 FPS | **6.192 FPS** | -1.86% |
| RealESRGAN_x2plus | 1080p | 2.754 FPS | **2.782 FPS** | 2.778 FPS | -1.01% |
| SPAN | 720p | 54.517 FPS | 55.498 FPS | **55.583 FPS** | -1.92% |
| SPAN | 1080p | **25.505 FPS** | 25.074 FPS | 25.232 FPS | +1.08% |

All four rows are within the predeclared +/-5% parity band. The measured
revision includes the torch-free runtime, streaming mux, and adaptive two-stage
tuning search. Both external implementations receive the same stream range,
runtime-default VapourSynth threads, automatic vspipe requests, and CUDA Graph
probe policy.

Absolute throughput is 1.7-3.0% lower than the superseded tuned snapshot across
all three implementations. The current session drew more average board power
and repeatedly reached the same 350 W software power cap, but peaked at 58 C
versus 65 C previously and recorded no thermal throttle reason. The evidence
therefore supports a shared cross-session or physical-GPU shift, not a
`trtvideo` regression or a thermal-throttling claim. Headline comparisons use
only the rotated measurements from the current physical GPU; old and current
absolute results are not pooled.

### Tuned Stream Sweep

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/tuned-sweep-dark.svg">
  <img alt="Tuned TensorRT stream-count sweep for RealESRGAN and SPAN at 720p and 1080p" src="figures/tuned-sweep-light.svg">
</picture>

The lines show one-run, 300-frame reconnaissance measurements. Rings mark the
profiles selected after full 1000-frame confirmation; the dashed `trtvideo`
reference is the independent final-campaign median, not a search result. Search
starts at one stream, stops after two confirmed declines greater than 1%, or
reaches the declared eight-stream boundary. The isolated RealESRGAN 1080p
eight-stream probes reached a reproducibly hashed CUDA out-of-memory ceiling;
the graph marks that limit rather than inventing an FPS point.

### Selected Profiles

| Workload | Input | vs-mlrt winner | VSGAN winner |
|---|---|---|---|
| RealESRGAN_x2plus | 720p | streams 2 | streams 2 |
| RealESRGAN_x2plus | 1080p | streams 2 | streams 2 |
| SPAN | 720p | streams 4 | streams 4 |
| SPAN | 1080p | streams 5 | streams 5 |

Every selected profile uses automatic vspipe requests, runtime-default
VapourSynth threads, and CUDA Graph disabled. Selection chooses the lowest
stream count within 1% of confirmed peak throughput, which is deliberately
favorable to the external implementation's CPU and VRAM use. The complete
reconnaissance curves, confirmation suites, stop reasons, and resource-ceiling
evidence are retained in [`tuned.json`](tuned.json).

### Intra-Session Reproducibility

The selected external profiles were measured independently during confirmation
and again in the final rotated campaign. Their medians agree within 0.44%:

| Workload | Input | vs-mlrt final vs confirmation | VSGAN final vs confirmation |
|---|---|---:|---:|
| RealESRGAN_x2plus | 720p | +0.11% | +0.24% |
| RealESRGAN_x2plus | 1080p | +0.10% | +0.15% |
| SPAN | 720p | +0.14% | +0.02% |
| SPAN | 1080p | -0.43% | +0.19% |

This is a same-session harness control, not another product comparison.
`trtvideo` is excluded because it is not a tuning candidate and therefore has
no confirmation-stage run.

### Resource Medians

CPU is attributed to the measured child-process tree through
`getrusage(RUSAGE_CHILDREN)`, not to total host activity.

| Workload | Input | Implementation | FPS | CPU cores | GPU util | Power | J/frame | Peak VRAM | Bitrate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RealESRGAN | 720p | trtvideo | 6.078 | **1.009** | 98.58% | 346.00 W | 56.93 | **2231.7 MiB** | 32.714 Mbps |
| RealESRGAN | 720p | vs-mlrt | 6.151 | 2.125 | 98.16% | 344.97 W | 56.04 | 3764.4 MiB | 35.042 Mbps |
| RealESRGAN | 720p | VSGAN | **6.192** | 2.126 | 98.14% | 344.81 W | **55.69** | 3762.4 MiB | 35.042 Mbps |
| RealESRGAN | 1080p | trtvideo | 2.754 | **1.005** | 99.38% | 347.75 W | 126.28 | **4264.1 MiB** | 56.651 Mbps |
| RealESRGAN | 1080p | vs-mlrt | **2.782** | 2.131 | 99.23% | 347.15 W | **124.81** | 7625.1 MiB | 60.200 Mbps |
| RealESRGAN | 1080p | VSGAN | 2.778 | 2.127 | 99.17% | 347.12 W | 124.92 | 7623.1 MiB | 60.200 Mbps |
| SPAN | 720p | trtvideo | 54.517 | **0.556** | 89.86% | 325.86 W | 5.98 | **1493.7 MiB** | 32.806 Mbps |
| SPAN | 720p | vs-mlrt | 55.498 | 5.402 | 90.98% | 322.28 W | **5.81** | 3936.4 MiB | 35.533 Mbps |
| SPAN | 720p | VSGAN | **55.583** | 5.413 | 92.68% | 323.18 W | 5.81 | 3934.4 MiB | 35.533 Mbps |
| SPAN | 1080p | trtvideo | **25.505** | **0.470** | 95.18% | 336.94 W | 13.20 | **2649.7 MiB** | 56.221 Mbps |
| SPAN | 1080p | vs-mlrt | 25.074 | 6.486 | 93.40% | 331.74 W | 13.24 | 9905.1 MiB | 60.466 Mbps |
| SPAN | 1080p | VSGAN | 25.232 | 6.459 | 94.41% | 332.49 W | **13.16** | 9903.1 MiB | 60.466 Mbps |

`trtvideo` reaches parity while using roughly half the CPU and substantially
less VRAM on RealESRGAN. On SPAN it uses about 0.5 CPU cores instead of
5.4-6.5 and 1.5-2.6 GiB VRAM instead of 3.8-9.7 GiB.

### Same Throughput, Lower Resource Use

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/throughput-resources-dark.svg">
  <img alt="Attributed CPU and peak VRAM for trtvideo versus the fastest external implementation at equivalent throughput" src="figures/throughput-resources-light.svg">
</picture>

Each row compares `trtvideo` with the fastest external implementation for that
workload on linear scales; the annotation beside the bars reports the
end-to-end FPS difference for the same pair.

## Upstream-Default Results

`vs-mlrt` uses automatic vspipe requests, one TensorRT stream,
runtime-default VapourSynth threads, and no CUDA Graph. VSGAN uses automatic
requests, four TensorRT streams, four VapourSynth threads, and no CUDA Graph.
`trtvideo` uses its production GPU-resident path unchanged.

| Workload | Input | trtvideo | vs-mlrt | VSGAN | Project vs fastest external |
|---|---|---:|---:|---:|---:|
| RealESRGAN_x2plus | 720p | **6.204 FPS** | 5.721 FPS | 6.201 FPS | +0.06% |
| RealESRGAN_x2plus | 1080p | **2.815 FPS** | 2.455 FPS | 2.811 FPS | +0.14% |
| SPAN | 720p | **56.160 FPS** | 26.835 FPS | 50.890 FPS | +10.36% |
| SPAN | 1080p | **26.273 FPS** | 9.514 FPS | 21.734 FPS | +20.88% |

The RealESRGAN rows are parity. The lightweight SPAN workload exposes the
scheduling and frame-transport cost of the upstream defaults: the project is
10.36% faster at 720p and 20.88% faster at 1080p.

This retained class predates the adaptive tuned measurement. It remains useful
for documenting upstream behavior, but its values are not used as a control
signal for the newer tuned session.

## Quality Gates

All winner profiles passed model-space validation on frames 0, 499, and 999.
Product-output metrics compare all 1000 decoded frames against `trtvideo`.

| Workload | Input | Candidate | PSNR | SSIM | Status |
|---|---|---|---:|---:|---|
| RealESRGAN_x2plus | 720p | vs-mlrt / VSGAN | 49.125 dB | 0.994492 | valid |
| RealESRGAN_x2plus | 1080p | vs-mlrt / VSGAN | 50.362 dB | 0.995249 | valid |
| SPAN | 720p | vs-mlrt / VSGAN | 48.898 dB | 0.995706 | valid |
| SPAN | 1080p | vs-mlrt / VSGAN | 50.291 dB | 0.996216 | valid |

VSGAN uses the same ONNX and weights but a separately built TensorRT 10.16
engine because its runtime cannot load the project's TensorRT 11 engine.

The identical vs-mlrt and VSGAN rows are expected, not reused measurements.
Both wrappers execute the `libvstrt.so` plugin through the same VapourSynth
graph and encoder contract. The quality jobs used separate output directories
and produced different capture manifests, run manifests, container image IDs,
and engine SHA-256 values. Despite that independent provenance, all six
model-space tensor SHA-256 values and the final MP4 SHA-256 match for each
workload. The retained PSNR/SSIM logs and visual crops are also byte-identical.

| Workload | Input | Candidate tensor-set SHA-256 | Candidate MP4 SHA-256 |
|---|---|---|---|
| RealESRGAN_x2plus | 720p | `fc16fa15a622e63d...` | `123bfd2d4c333b5e...` |
| RealESRGAN_x2plus | 1080p | `cc67551013893833...` | `59e6780ad2be4d55...` |
| SPAN | 720p | `8e7a291bdc7e3b68...` | `a0bb9fdc12530d90...` |
| SPAN | 1080p | `751b765d34995af5...` | `50244984f6b8110a...` |

The full fingerprints and provenance assertions are retained in both result
JSON files. The tensor-set fingerprint is the SHA-256 of the ordered
`stage<TAB>frame_index<TAB>artifact_sha256<LF>` records from the capture
manifest.

The repeated model-space minimum across RealESRGAN and SPAN at the same input
resolution is also expected. In each case the minimum occurs at input frame
499, before model inference. Both workloads use the same source clip and
preprocessing, so that input tensor and its comparison metric are identical.

## Diagnostics

### TensorRT Ceiling

`trtexec` is an inference-only diagnostic, not a product competitor.

| Workload | Input | trtexec median | Co-measured trtvideo | Pipeline efficiency |
|---|---|---:|---:|---:|
| RealESRGAN_x2plus | 720p | 6.283 QPS | 6.204 FPS | 98.73% |
| RealESRGAN_x2plus | 1080p | 2.829 QPS | 2.813 FPS | 99.42% |
| SPAN | 720p | 60.951 QPS | 56.187 FPS | 92.18% |
| SPAN | 1080p | 27.553 QPS | 26.229 FPS | 95.20% |

CUDA Graph and data transfers were disabled for the TensorRT ceiling.
This table belongs to the 2026-07-30 diagnostic snapshot and is not recomputed
by combining its `trtexec` values with the newer tuned campaign.

### Nsight Systems

A 120-frame SPAN 1080p trace confirms the GPU-resident architecture:

- CUDA kernel intervals cover 99.37% of the frame-loop interval;
- NVDEC and NVENC workloads overlap CUDA kernels by 93.55% and 99.54%;
- no H2D or D2H copy occurs during the measured frame loop;
- D2D copies average 14.83 MiB and 0.043 ms per frame.

Profiler overhead makes trace FPS non-publishable. The trace supports the
architecture claim rather than adding another throughput result.

## Encoding Note

All products use the same H.264 P4/HQ single-pass CBR contract, GOP 24, zero
B-frames, and disabled lookahead/AQ. The FFmpeg NVENC path inserts filler NAL
units for stricter CBR while PyNvVideoCodec does not, producing the visible
bitrate difference. Full decode, timestamps, color metadata, and quality gates
remain valid.

## Published Data

[`index.json`](index.json) records result composition, revision, and SHA-256.
[`upstream-default.json`](upstream-default.json), [`tuned.json`](tuned.json),
and [`diagnostics.json`](diagnostics.json) retain per-run FPS, resources,
profiles, quality summaries, assets, and raw-evidence hashes.

All SVG figures are generated from those committed JSON files with
`make -C benchmarks figures`; `make -C benchmarks figures-check` verifies
byte-for-byte reproducibility.

MP4 outputs, FP32 tensor captures, NVML time series, engines, models, event
logs, and profiler traces remain outside Git. A second live-action confirmation
clip remains future work.
