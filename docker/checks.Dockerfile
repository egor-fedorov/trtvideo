# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.15

FROM ${UV_IMAGE} AS uv
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    VIRTUAL_ENV=/opt/trtvideo \
    TRTVIDEO_IMAGE_VARIANT=development
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv venv "${VIRTUAL_ENV}" && \
    uv export --frozen --no-emit-project --group dev \
        --output-file /tmp/requirements.txt && \
    uv pip install --python "${VIRTUAL_ENV}" -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY README.md ./
COPY src/ src/
COPY benchmarks/ benchmarks/
COPY tests/ tests/
COPY --chmod=755 benchmarks/bin/benchmark-trtvideo /usr/local/bin/benchmark-trtvideo
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${VIRTUAL_ENV}" --no-deps .
