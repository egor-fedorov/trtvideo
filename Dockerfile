# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tensorrt:26.03-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg

ARG UV_VERSION=0.8.15
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "uv==${UV_VERSION}"

ARG INSTALL_DEV=0
COPY pyproject.toml uv.lock ./
# Keep sync inexact because the TensorRT base image preinstalls runtime packages
# that are intentionally outside this project's dependency metadata.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    if [ "$INSTALL_DEV" = "1" ]; then \
        uv sync --frozen --inexact --no-install-project --extra docker --group dev; \
    else \
        uv sync --frozen --inexact --no-dev --no-install-project --extra docker; \
    fi

COPY upscaler/ upscaler/
COPY benchmark.py inference.py inference_gpu.py ./
COPY tools/ tools/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --inexact --no-editable --only-install-project

ENTRYPOINT []
