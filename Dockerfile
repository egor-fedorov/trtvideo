# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/tensorrt:26.04-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    VIRTUAL_ENV=/opt/upscaler \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg

ARG UV_VERSION=0.8.15
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "uv==${UV_VERSION}"

ARG INSTALL_DEV=0
COPY pyproject.toml uv.lock ./
# Use a dedicated venv because the base image's /usr Python is externally managed.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv venv "${VIRTUAL_ENV}" --system-site-packages && \
    if [ "$INSTALL_DEV" = "1" ]; then \
        uv export --frozen --no-emit-project --extra docker --group dev \
            --output-file /tmp/requirements.txt; \
    else \
        uv export --frozen --no-emit-project --no-dev --extra docker \
            --output-file /tmp/requirements.txt; \
    fi && \
    uv pip install --python "${VIRTUAL_ENV}" -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY upscaler/ upscaler/
COPY benchmark.py inference.py inference_gpu.py ./
COPY tools/ tools/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${VIRTUAL_ENV}" --no-deps .

ENTRYPOINT []
