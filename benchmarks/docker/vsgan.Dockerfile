# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0
ARG BASE_IMAGE=docker.io/styler00dollar/vsgan_tensorrt@sha256:1f23b8b43864021fb5a9e795c72e0a51b2bba568e6bbc24be175f924b828aaef
FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VCS_REF=unknown
ARG VCS_DIRTY=unknown
ARG NVIDIA_ML_PY_VERSION=13.610.43
ARG PYTHON_VERSION=3.12.3
ARG BENCHMARK_VENV=/opt/vsgan-benchmark
ARG FFMPEG_VERSION=7:6.1.1-3ubuntu5

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends "ffmpeg=${FFMPEG_VERSION}"

# Use the normalized encoder instead of upstream FFmpeg built for NVENC API 13.1.
ENV PATH="/usr/bin:${PATH}"

RUN test "$(od -An -tx1 -N4 /usr/local/bin/vspipe | tr -d ' \n')" = "7f454c46" && \
    test "$(command -v ffmpeg)" = "/usr/bin/ffmpeg" && \
    test "$(command -v ffprobe)" = "/usr/bin/ffprobe" && \
    test "$(dpkg-query -W -f='${Version}' ffmpeg)" = "${FFMPEG_VERSION}" && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_nvenc' && \
    vspipe --version && \
    ffprobe -version

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed

# Keep the runner isolated without activating its venv for embedded VSScript.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv python install "${PYTHON_VERSION}" && \
    uv venv "${BENCHMARK_VENV}" --python "${PYTHON_VERSION}" && \
    uv pip install --python "${BENCHMARK_VENV}" \
        "nvidia-ml-py==${NVIDIA_ML_PY_VERSION}" && \
    "${BENCHMARK_VENV}/bin/python3" -c "import pynvml" && \
    vspipe --version

WORKDIR /app
COPY src/ src/
COPY benchmarks/ benchmarks/

ENV PYTHONPATH=/app/src \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TRTVIDEO_BASE_IMAGE=${BASE_IMAGE} \
    TRTVIDEO_VSGAN_FFMPEG_PACKAGE=${FFMPEG_VERSION} \
    TRTVIDEO_BUILD_REVISION=${VCS_REF} \
    TRTVIDEO_BUILD_DIRTY=${VCS_DIRTY}

LABEL org.opencontainers.image.source="https://github.com/styler00dollar/VSGAN-tensorrt-docker" \
      org.opencontainers.image.revision="f4c06ed08e0d09952cf8671ec453f53c029c2158"

ENTRYPOINT []
