# Licensing And Redistribution Audit

This document records the public distribution boundary and release inventory
reviewed on 2026-08-13. It is a technical record, not legal advice. The evidence
must be reviewed again when the TensorRT base, Ubuntu release, direct
dependencies, model assets, or benchmark media change.

## Distribution Boundary

| Material | Distributed by this repository or GHCR workflow? | Governing terms |
|---|---|---|
| `trtvideo` source and documentation | Yes | Apache-2.0 |
| Production container | Yes, for versioned releases | Mixed; see `THIRD_PARTY_NOTICES.md` and the release SBOM |
| `model-tools` and benchmark containers | No | Local/internal targets only |
| Model weights, ONNX, TensorRT engines | No | Upstream model terms |
| Input media and raw benchmark artifacts | No | Upstream media terms |
| Compact benchmark JSON and generated SVGs | Yes | Repository Apache-2.0; underlying measurements retain recorded provenance |

The production image is intentionally narrower than the local toolchain. It
contains `trtvideo`, `build-engine`, TensorRT, CUDA Python bindings, CV-CUDA,
PyNvVideoCodec, ONNX, and FFmpeg. PyTorch, torchvision, Spandrel, ONNX Script,
and ONNX conversion tools exist only in the non-published `model-tools` and
benchmark targets.

## NVIDIA Components And Benchmark Publication

The Dockerfile pins the TensorRT NGC base by digest. The base-provided NVIDIA
notices are preserved in every derived target. At audit time, NVIDIA's AI
Product Terms classify publicly available no-cost NGC software as Community
Products and expressly permit distribution as part of a customer product,
subject to their distribution requirements. Proprietary SDK components such as
CUDA and TensorRT remain restricted to NVIDIA platforms. Consumers must review
and accept the current:

- [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/);
- [Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

The NVIDIA agreement normally restricts publication of benchmark and
performance data. NVIDIA's published
[container benchmarking policy](https://docs.nvidia.com/nvidia-containers-benchmarking.pdf)
lists `nvcr.io/nvidia/tensorrt:*` among the containers whose benchmark,
regression, and performance data may be published without separate permission.
The repository's TensorRT measurements therefore remain tied to that named
container family and exact image provenance.

## Model And Media Assets

The tools download these assets only into ignored local directories:

| Asset | Use | Recorded license | Redistribution policy |
|---|---|---|---|
| RealESRGAN_x2plus v0.2.1 weights | Demo and benchmark | BSD-3-Clause project license | Download from the pinned upstream release; do not add to an image or git |
| 2xLiveActionV1_SPAN weights | Benchmark only | CC-BY-NC-SA-4.0 | Non-commercial benchmark asset; download locally and never publish with the image |
| Madrid-2021-05-06 source | Benchmark input | CC0-1.0 | Download locally; prepared clips and raw media remain excluded |

The workload manifests are the machine-readable sources of truth for URLs,
hashes, sizes, license references, and attribution. Generated ONNX and TensorRT
engines are also excluded because they derive from the selected weights and are
GPU/TensorRT specific.

## Published Image Contract

Every versioned image is built from a clean release tag and only from the
`production` target. The image excludes repository-local models, media, demo
files, benchmark artifacts, PyTorch model tooling, and benchmark tooling. Its
TensorRT base is selected by immutable digest, and the base-provided NVIDIA plus
Ubuntu package notices remain intact.

The registry manifest carries an SBOM and build provenance. A signed GitHub
attestation binds the image digest to the release revision, and the immutable
digest is recorded with the GitHub release. The release evidence must identify
the exact FFmpeg/x264 package versions and corresponding Ubuntu source packages.
`make check` and the GPU acceptance in `docs/TESTING.md` apply before a release.

No automated check can establish that a use or redistribution is legally
permitted. Automation records composition and provenance so that a legal
decision can be made against concrete evidence rather than an assumed
dependency list.
