# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:26.06-py3
FROM ${BASE_IMAGE} AS runtime

ARG BASE_IMAGE

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    VIRTUAL_ENV=/opt/ai-media-enhancer \
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

ARG VCS_REF=unknown
ARG VCS_DIRTY=unknown
ENV AI_MEDIA_BASE_IMAGE="${BASE_IMAGE}" \
    AI_MEDIA_BUILD_REVISION="${VCS_REF}" \
    AI_MEDIA_BUILD_DIRTY="${VCS_DIRTY}"
LABEL org.opencontainers.image.revision="${VCS_REF}"

COPY ai_media/ ai_media/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${VIRTUAL_ENV}" --no-deps .

ENTRYPOINT []

FROM runtime AS benchmark

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv export --frozen --no-emit-project --no-dev --extra benchmark \
        --output-file /tmp/benchmark-requirements.txt && \
    uv pip install --python "${VIRTUAL_ENV}" -r /tmp/benchmark-requirements.txt && \
    rm /tmp/benchmark-requirements.txt

COPY benchmarks/ benchmarks/
COPY --chmod=755 benchmarks/bin/benchmark-upscale /usr/local/bin/benchmark-upscale

FROM runtime AS production
