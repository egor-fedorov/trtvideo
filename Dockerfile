# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:26.06-py3@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3
FROM ${BASE_IMAGE} AS runtime

ARG BASE_IMAGE

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    VIRTUAL_ENV=/opt/trtvideo \
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

ARG BUILD_DATE=unknown
ARG VERSION=dev
ARG VCS_REF=unknown
ARG VCS_DIRTY=unknown
ENV TRTVIDEO_BASE_IMAGE="${BASE_IMAGE}" \
    TRTVIDEO_BUILD_REVISION="${VCS_REF}" \
    TRTVIDEO_BUILD_DIRTY="${VCS_DIRTY}"
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.description="GPU-resident TensorRT video processing" \
      org.opencontainers.image.documentation="https://github.com/egor-fedorov/trtvideo#readme" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/egor-fedorov/trtvideo" \
      org.opencontainers.image.title="trtvideo" \
      org.opencontainers.image.url="https://github.com/egor-fedorov/trtvideo" \
      org.opencontainers.image.version="${VERSION}"

COPY README.md LICENSE ./
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/doc/trtvideo/
COPY docs/LICENSING.md /usr/share/doc/trtvideo/LICENSING.md
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${VIRTUAL_ENV}" --no-deps .

ENTRYPOINT []

FROM runtime AS model-tools

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --active --inexact --no-install-project \
        --extra docker --extra export

# Target identity changes per release without invalidating the dependency layer.
ARG IMAGE_REFERENCE=trtvideo:model-tools
ENV TRTVIDEO_IMAGE_REF="${IMAGE_REFERENCE}" \
    TRTVIDEO_IMAGE_VARIANT=model-tools

FROM model-tools AS benchmark

ENV TRTVIDEO_IMAGE_VARIANT=benchmark

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --active --inexact --no-install-project \
        --extra docker --extra export --extra benchmark

COPY benchmarks/ benchmarks/
COPY --chmod=755 benchmarks/bin/benchmark-trtvideo /usr/local/bin/benchmark-trtvideo

FROM runtime AS production

ARG IMAGE_REFERENCE=trtvideo:latest
ENV TRTVIDEO_IMAGE_REF="${IMAGE_REFERENCE}" \
    TRTVIDEO_IMAGE_VARIANT=production

# PyTorch model export and ONNX rewriting are confined to model-tools.
RUN rm -f "${VIRTUAL_ENV}/bin/export-onnx" \
          "${VIRTUAL_ENV}/bin/prepare-onnx"
