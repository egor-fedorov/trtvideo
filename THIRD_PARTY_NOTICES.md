# Third-Party Notices

This notice describes the principal third-party components in the `trtvideo`
container targets. It is an inventory, not a replacement for the license text
shipped with each component and not legal advice.

The `trtvideo` source and documentation are Apache-2.0. The complete image has
no single project-wide SPDX license because the base image, operating-system
packages, and Python packages retain their own terms. Model weights and media
are not included.

## Production Image

### NVIDIA TensorRT Base

The production target derives from the immutable TensorRT base image recorded
in `Dockerfile`. TensorRT, CUDA, `cuda-bindings`, and other NVIDIA components
retain their NVIDIA or component-specific terms. The base image states that it
is governed by the:

- [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/);
- [Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

The original NVIDIA notices and TensorRT OSS notices remain in the image under
`/opt/nvidia/`, `/workspace/license.txt`, and `/workspace/tensorrt/oss/`.
NVIDIA's proprietary SDK components are licensed only for systems with NVIDIA
platforms. Users and redistributors are responsible for complying with the
applicable NVIDIA terms.

### Ubuntu And FFmpeg

The base is Ubuntu 24.04 and the image installs the Ubuntu `ffmpeg` package.
Ubuntu's default FFmpeg binaries include GPL components; the package copyright
record describes the resulting command-line binaries as GPL-2.0-or-later and
some library variants as effectively GPL-3.0-or-later. The dependency closure
also contains the GPL-2.0-or-later `libx264` package.

Exact package versions are recorded in each release SBOM. Package copyright
and license texts remain under `/usr/share/doc/<package>/copyright` and
`/usr/share/common-licenses/`. Corresponding Ubuntu source packages are
available from the [Ubuntu source package archive](https://packages.ubuntu.com/source/noble/ffmpeg)
and the [Ubuntu package snapshot service](https://snapshot.ubuntu.com/), keyed
by the versions in that SBOM.

FFmpeg and the Python application are separate programs communicating through
process and pipe interfaces. The GPL terms still apply to redistribution of the
FFmpeg and x264 binaries themselves.

### Direct Python Components

The production target installs these direct Python components in addition to
the packages already present in the TensorRT base:

| Component | License recorded by the package |
|---|---|
| CUDA Python bindings (`cuda-bindings`) | `LicenseRef-NVIDIA-SOFTWARE-LICENSE` |
| CV-CUDA (`cvcuda-cu12`) | Apache-2.0 |
| PyNvVideoCodec | MIT |
| ONNX | Apache-2.0 |

Their complete package metadata and license files remain in the Python
installation. Transitive packages and exact versions are enumerated by the
release SBOM attached to the GHCR image manifest.

## Local Model-Tools And Benchmark Targets

The `model-tools` target additionally installs CPU-only PyTorch and torchvision
(BSD-family), Spandrel (MIT), ONNX Script (MIT), and ONNX Converter Common
(MIT). The `benchmark` target additionally installs its diagnostic packages.
The release workflow does not publish either target to GHCR.

## Assets Not Included

Model weights, exported ONNX files, TensorRT engines, input videos, benchmark
outputs, and raw benchmark artifacts are not included in the source repository
or published production image. Their source, license, and attribution records
are documented in `docs/LICENSING.md` and the benchmark workload manifests.
