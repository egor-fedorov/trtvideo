# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0
ARG BASE_IMAGE=ghcr.io/k4yt3x/video2x@sha256:e21b6893269b4cb6f5603802726fd7537be241f6b39217b73530478861acbca1
FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VCS_REF=unknown
ARG VCS_DIRTY=unknown
ARG NVIDIA_ML_PY_VERSION=13.610.43
ARG PYTHON_VERSION=3.12.3

COPY --from=uv /uv /uvx /usr/local/bin/

ENV VIRTUAL_ENV=/opt/video2x-benchmark \
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

LABEL org.opencontainers.image.source="https://github.com/k4yt3x/video2x" \
      org.opencontainers.image.revision="a96bda9b4d79616cc6b71b94e6945146b5b4d509"

ENTRYPOINT []
