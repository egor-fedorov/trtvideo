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

RUN test "$(od -An -tx1 -N4 /usr/local/bin/vspipe | tr -d ' \n')" = "7f454c46" && \
    vspipe --version

COPY --from=uv /uv /uvx /usr/local/bin/

ENV VIRTUAL_ENV=/opt/vsgan-benchmark \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv python install "${PYTHON_VERSION}" && \
    uv venv "${VIRTUAL_ENV}" --python "${PYTHON_VERSION}" && \
    uv pip install --python "${VIRTUAL_ENV}" \
        "nvidia-ml-py==${NVIDIA_ML_PY_VERSION}"

WORKDIR /app
COPY ai_media/ ai_media/
COPY benchmarks/ benchmarks/

ENV PYTHONPATH=/app \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    AI_MEDIA_BASE_IMAGE=${BASE_IMAGE} \
    AI_MEDIA_BUILD_REVISION=${VCS_REF} \
    AI_MEDIA_BUILD_DIRTY=${VCS_DIRTY}

LABEL org.opencontainers.image.source="https://github.com/styler00dollar/VSGAN-tensorrt-docker" \
      org.opencontainers.image.revision="f4c06ed08e0d09952cf8671ec453f53c029c2158"

ENTRYPOINT []
